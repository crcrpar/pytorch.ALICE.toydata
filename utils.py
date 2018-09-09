import torch
import torch.nn.functional as F


def sigmoid_x_entropy(input, real_fake=True, _reduce=True):
    # Ref: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
    labels = torch.ones_like(input) if real_fake else torch.zeros_like(input)
    zeros = torch.zeros_like(input)
    x_ent = torch.max(input, zeros) - input * labels + torch.log(1 + input.abs().exp())
    if _reduce:
        return x_ent.mean()
    else:
        return x_ent


def forward(device, adversarial, x, z,
            gen, inf, dis, dis_x=None, dis_z=None,
            return_all=False):
    assert isinstance(adversarial, bool)
    p_x, q_z = gen(z), inf(x)
    dis_gen_logit, dis_inf_logit = dis(p_x, z), dis(x, q_z)
    dis_gen_loss = sigmoid_x_entropy(dis_gen_logit, False)
    dis_inf_loss = sigmoid_x_entropy(dis_inf_logit, True)
    dis_loss_xz = dis_inf_loss + dis_gen_loss  # opt_dis
    dis_gen_loss2 = sigmoid_x_entropy(dis_gen_logit, True)
    dis_inf_loss2 = sigmoid_x_entropy(dis_inf_logit, False)
    gen_loss_xz = dis_gen_loss2 + dis_inf_loss2  # opt_gen_inf

    x_rec, z_rec = gen(q_z), inf(p_x)
    if adversarial:
        disx_real, disx_fake = dis_x(x, x), dis_x(x, x_rec)
        disz_real, disz_fake = dis_z(z, z), dis_z(z, z_rec)
        disx_real_loss = sigmoid_x_entropy(disx_real, True)
        disx_fake_loss = sigmoid_x_entropy(disx_fake, False)
        disz_real_loss = sigmoid_x_entropy(disz_real, True)
        disz_fake_loss = sigmoid_x_entropy(disz_fake, False)
        disx_real_loss2 = sigmoid_x_entropy(disx_real, False)
        disx_fake_loss2 = sigmoid_x_entropy(disx_fake, True)
        disz_real_loss2 = sigmoid_x_entropy(disz_real, False)
        disz_fake_loss2 = sigmoid_x_entropy(disz_fake, True)

    # Loss for the parameters managed by `opt_dis`
    dis_loss_opt = dis_loss_xz
    if adversarial:
        disx_loss = disx_real_loss + disx_fake_loss
        disz_loss = disz_real_loss + disz_fake_loss
        dis_loss_opt += disx_loss + disz_loss

    # Loss for the parameters managed by `opt_gen_inf`
    if adversarial:
        cost_x = disx_real_loss2 + disx_fake_loss2
        cost_z = disz_real_loss2 + disz_fake_loss2
    else:
        cost_x = F.mse_loss(x_rec, x)
        cost_z = F.mse_loss(z_rec, z)
    gen_loss_opt = gen_loss_xz + 0.1 * cost_x + 0.1 * cost_z

    if return_all:
        return dis_loss_opt, gen_loss_opt, cost_x, cost_z
    return dis_loss_opt, gen_loss_opt
