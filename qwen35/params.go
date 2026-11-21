package qwen35

import (
	"fmt"

	"github.com/guregodevo/janis/qwen"
)

// Params lists the weights an edit may replace, by checkpoint name. The
// embedding is not one (its table is read by row, not multiplied), nor is
// A_log: the model keeps -exp(A_log), folded at load.
func (m *Model) Params() []qwen.Param {
	var ps []qwen.Param
	for i, bl := range m.Blocks {
		lp := fmt.Sprintf("%smodel.layers.%d", prefix, i)
		ps = append(ps,
			qwen.NormParam(lp+".input_layernorm", bl.InputNorm),
			qwen.NormParam(lp+".post_attention_layernorm", bl.PostNorm),
			qwen.QuantParam(lp+".mlp.gate_proj", bl.FFN.Gate),
			qwen.QuantParam(lp+".mlp.up_proj", bl.FFN.Up),
			qwen.QuantParam(lp+".mlp.down_proj", bl.FFN.Down),
		)
		if d := bl.Linear; d != nil {
			la := lp + ".linear_attn"
			ps = append(ps,
				qwen.QuantParam(la+".in_proj_qkv", d.InQKV),
				qwen.QuantParam(la+".in_proj_z", d.InZ),
				qwen.QuantParam(la+".in_proj_a", d.InA),
				qwen.QuantParam(la+".in_proj_b", d.InB),
				qwen.QuantParam(la+".out_proj", d.OutProj),
				qwen.Param{Name: la + ".conv1d.weight", Plain: &d.ConvW},
				qwen.Param{Name: la + ".dt_bias", Plain: &d.DtBias},
				qwen.Param{Name: la + ".norm.weight", Plain: &d.NormW},
			)
		}
		if a := bl.Attn; a != nil {
			sa := lp + ".self_attn"
			ps = append(ps,
				qwen.QuantParam(sa+".q_proj", a.QProj),
				qwen.QuantParam(sa+".k_proj", a.KProj),
				qwen.QuantParam(sa+".v_proj", a.VProj),
				qwen.QuantParam(sa+".o_proj", a.OProj),
				qwen.NormParam(sa+".q_norm", a.QNorm),
				qwen.NormParam(sa+".k_norm", a.KNorm),
			)
		}
	}
	ps = append(ps, qwen.NormParam(prefix+"model.norm", m.Norm))
	if m.LMHead != nil {
		ps = append(ps, qwen.QuantParam(prefix+"lm_head", m.LMHead))
	}
	return ps
}
