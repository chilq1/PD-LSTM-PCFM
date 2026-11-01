import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PhysicalHiddenModulation(nn.Module):
    """Apply PCFM to readout features using p = [dBx/S_Bx, dBy/S_By]."""

    def __init__(self, hidden_dim, feat_dim=2, gate_hidden=64,
                 modulation_scale=0.5):
        super().__init__()
        if not 0.0 < modulation_scale < 1.0:
            raise ValueError("modulation_scale must lie between 0 and 1")
        self.modulation_scale = float(modulation_scale)
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim + feat_dim, gate_hidden),
            nn.Tanh(),
            nn.Linear(gate_hidden, hidden_dim),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, hidden_state, phys_feats):
        """Return modulated features and factors without changing LSTM states."""
        combined = torch.cat([hidden_state, phys_feats], dim=1)
        r = self.gate_net(combined)
        rho = 1.0 + self.modulation_scale * torch.tanh(r)
        return hidden_state * rho, rho


class BaselineModel(nn.Module):
    """Two-layer LSTM with readout-only PCFM and a Gram-based tensor output."""

    def __init__(self, scaler_info, hidden_dim=128, output_dim=6, num_layers=2):
        super().__init__()
        if output_dim < 6:
            raise ValueError("output_dim must be >= 6 for the SPD tensor construction")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm_input_size = 5
        self.phys_feat_dim = 2

        self.cells = nn.ModuleList()
        for i in range(num_layers):
            layer_input_dim = self.lstm_input_size if i == 0 else hidden_dim
            self.cells.append(nn.LSTMCell(layer_input_dim, hidden_dim))

        self.hidden_modulation = PhysicalHiddenModulation(
            hidden_dim, self.phys_feat_dim, gate_hidden=64, modulation_scale=0.5
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim + self.phys_feat_dim, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, output_dim),
        )

        p_scales = scaler_info['physics_scales']
        for name in ('S_Bx', 'S_By', 'S_Hx', 'S_Hy', 'mid_Hx', 'mid_Hy'):
            self.register_buffer(name, torch.tensor(p_scales[name], dtype=torch.float32))

        self.scale_mx = math.sqrt(p_scales['S_Hx'] / p_scales['S_Bx'])
        self.scale_my = math.sqrt(p_scales['S_Hy'] / p_scales['S_By'])
        self.scale_asym = math.sqrt(
            (p_scales['S_Hx'] / p_scales['S_Bx']) *
            (p_scales['S_Hy'] / p_scales['S_By'])
        )

    @torch.jit.export
    def forward_step(self, x_curr: torch.Tensor, b_prev: torch.Tensor,
                     h_prev_list: List[torch.Tensor],
                     c_prev_list: Optional[List[torch.Tensor]] = None):
        """Construct Jdiff from current B, previous H, T and previous B.

        x_curr: [batch, 5], ordered as normalized Bx, By, previous Hx, Hy, T.
        b_prev: [batch, 2], containing previous Bx and By with the same normalization.
        PCFM uses p = B_norm - B_prev_norm = [dBx/S_Bx, dBy/S_By].
        Returns Jdiff [J11, J12, J21, J22] and unmodulated hidden/cell states.
        """
        if x_curr.dim() != 2 or x_curr.size(1) != 5:
            raise ValueError("x_curr must have shape [batch, 5]")
        if b_prev.dim() != 2 or b_prev.size(1) != 2 or b_prev.size(0) != x_curr.size(0):
            raise ValueError("b_prev must have shape [batch, 2]")
        phys_feats = x_curr[:, 0:2] - b_prev
        if c_prev_list is None:
            c_prev_list = [torch.zeros_like(h_prev_list[i]) for i in range(self.num_layers)]

        h_curr_list = torch.jit.annotate(List[torch.Tensor], [])
        c_curr_list = torch.jit.annotate(List[torch.Tensor], [])
        layer_input = x_curr
        for i, cell in enumerate(self.cells):
            h_curr, c_curr = cell(layer_input, (h_prev_list[i], c_prev_list[i]))
            h_curr_list.append(h_curr)
            c_curr_list.append(c_curr)
            layer_input = h_curr

        hidden_mod, _ = self.hidden_modulation(h_curr_list[-1], phys_feats)
        output = self.readout(torch.cat([hidden_mod, phys_feats], dim=1))

        m11 = output[:, 0:1] * self.scale_mx
        m12 = output[:, 1:2] * self.scale_my
        m21 = output[:, 2:3] * self.scale_mx
        m22 = output[:, 3:4] * self.scale_my
        gamma1 = output[:, 4:5]
        gamma2 = output[:, 5:6]

        eps = 1e-8
        j11 = m11 ** 2 + m21 ** 2 + eps
        j22 = m12 ** 2 + m22 ** 2 + eps
        j_sym = (m11 * m12 + m21 * m22) * torch.tanh(gamma1)
        j_asym = gamma2 * self.scale_asym
        jac_phys = torch.cat([j11, j_sym + j_asym, j_sym - j_asym, j22], dim=1)
        return jac_phys, h_curr_list, c_curr_list

    def forward(self, input_seq: torch.Tensor):
        """Roll out H with physical dH = Jdiff dB and normalized H feedback.

        Input shape: [batch, steps, 7]. Columns: T_norm, Bx_norm, By_norm,
        Bx_prev_norm, By_prev_norm, Hx_prev_norm, Hy_prev_norm.
        Initial H is taken from the first row; later H inputs use predictions.
        Previous B defines each increment and must match the sampled trajectory.
        """
        if input_seq.dim() != 3 or input_seq.size(2) != 7:
            raise ValueError("input_seq must have shape [batch, steps, 7]")
        batch_size, seq_len, _ = input_seq.shape
        if seq_len == 0:
            raise ValueError("input_seq must contain at least one update step")
        h_curr_list = torch.jit.annotate(
            List[torch.Tensor],
            [input_seq.new_zeros((batch_size, self.hidden_dim))
             for _ in range(self.num_layers)],
        )
        c_curr_list = torch.jit.annotate(
            List[torch.Tensor],
            [input_seq.new_zeros((batch_size, self.hidden_dim))
             for _ in range(self.num_layers)],
        )
        outputs = torch.jit.annotate(List[Tuple[torch.Tensor, torch.Tensor]], [])
        curr_hx_norm = input_seq[:, 0, 5:6]
        curr_hy_norm = input_seq[:, 0, 6:7]
        curr_hx_phys = curr_hx_norm * self.S_Hx + self.mid_Hx
        curr_hy_phys = curr_hy_norm * self.S_Hy + self.mid_Hy

        for t in range(seq_len):
            x_step = torch.cat([
                input_seq[:, t, 1:3], curr_hx_norm, curr_hy_norm,
                input_seq[:, t, 0:1],
            ], dim=1)
            b_prev = input_seq[:, t, 3:5]
            delta_b_norm = input_seq[:, t, 1:3] - b_prev
            jac_phys, h_curr_list, c_curr_list = self.forward_step(
                x_step, b_prev, h_curr_list, c_curr_list
            )

            dbx_phys = delta_b_norm[:, 0:1] * self.S_Bx
            dby_phys = delta_b_norm[:, 1:2] * self.S_By
            curr_hx_phys = curr_hx_phys + jac_phys[:, 0:1] * dbx_phys + jac_phys[:, 1:2] * dby_phys
            curr_hy_phys = curr_hy_phys + jac_phys[:, 2:3] * dbx_phys + jac_phys[:, 3:4] * dby_phys
            curr_hx_norm = (curr_hx_phys - self.mid_Hx) / self.S_Hx
            curr_hy_norm = (curr_hy_phys - self.mid_Hy) / self.S_Hy
            outputs.append((torch.cat([curr_hx_norm, curr_hy_norm], dim=1), jac_phys))

        return outputs


class PhysicsInformedLoss(nn.Module):
    """Sum the separate Hx and Hy Huber losses, averaged over batch and steps."""

    def __init__(self, delta=1.0):
        super().__init__()
        if delta <= 0.0:
            raise ValueError("delta must be positive")
        self.huber_loss = nn.HuberLoss(delta=delta, reduction='mean')

    def forward(self, model_outputs, targets, return_dict=False):
        """Compare normalized H predictions with targets of shape [batch, steps, 2]."""
        h_pred_seq = torch.stack(
            [step[0] for step in model_outputs], dim=1
        )
        if h_pred_seq.shape != targets.shape:
            raise ValueError("targets must have shape [batch, steps, 2]")
        loss_hx = self.huber_loss(h_pred_seq[:, :, 0], targets[:, :, 0])
        loss_hy = self.huber_loss(h_pred_seq[:, :, 1], targets[:, :, 1])
        loss = loss_hx + loss_hy

        if return_dict:
            return loss, {
                "loss_hx": loss_hx.detach(),
                "loss_hy": loss_hy.detach(),
            }
        return loss
