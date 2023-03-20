import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence

from functions import vq, vq_st


def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class VAE(nn.Module):
    def __init__(self, input_dim, dim, z_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, z_dim * 2, 3, 1, 0),
            nn.BatchNorm2d(z_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_dim, dim, 3, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 5, 1, 0),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = Normal(mu, logvar.mul(.5).exp())
        p_z = Normal(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()

        x_tilde = self.decoder(q_z_x.rsample())
        return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar



class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

        self.codebook = VQEmbedding(K, dim)

        self.decoder = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out





class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(self, input_dim, dim):
        super(Encoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim),
            ResBlock(dim),
        )

    def forward(self, x):
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, dim, input_dim):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.fc(x)
##############################################################################################################################

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.BatchNorm2d(channel), #MY
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.BatchNorm2d(in_channel), #MY
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 16:
            blocks = [
                nn.Conv2d(in_channel, channel // 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//8),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
                #nn.ReLU(inplace=True),
            ]

        elif stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, apply_sigmoid = True):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 16:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 4, channel // 8, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 8, out_channel, 4, stride=2, padding=1),
            ])

        elif stride == 8:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, channel // 4, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 4, out_channel, 4, stride=2, padding=1),
            ])

        elif stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, 4, stride=2, padding=1),
            ])

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)
        if apply_sigmoid:
            self.last_layer = nn.Sigmoid()
        else:
            self.last_layer = nn.Identity()

    def forward(self, input):
        return self.last_layer(self.blocks(input))

class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, channel=128, n_res_block=2, n_res_channel=32, stride = 4, K=512):
        super().__init__()
        self.encoder = Encoder( input_dim, channel, n_res_block, n_res_channel, stride)#Encoder(input_dim, dim)
        self.quantize_conv = nn.Conv2d(channel, dim, 1)
        self.codebook = VQEmbedding(K, dim)
        self.decoder = Decoder(dim, input_dim, channel, n_res_block, n_res_channel, stride)

        self.apply(weights_init)

    def encode(self, x):
        pre_z_e_x = self.encoder(x)
        z_e_x = self.quantize_conv(pre_z_e_x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        pre_z_e_x = self.encoder(x)
        z_e_x = self.quantize_conv(pre_z_e_x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
################ monster
class VQVAEMonster(nn.Module):
    def __init__(self, in_channel=1, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=32, decay=0.99, #n_embed default 512
                 first_stride=4, second_stride=2):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=first_stride)
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=second_stride)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VQEmbedding(n_embed, embed_dim) #Quantize(embed_dim, n_embed)
        self.dec_t = Decoder(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=second_stride, apply_sigmoid = False)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VQEmbedding(n_embed, embed_dim) #Quantize(embed_dim, n_embed)
        self.upsample_t = Decoder(embed_dim, embed_dim, embed_dim, n_res_block, n_res_channel, stride=second_stride)
        self.dec = Decoder(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=first_stride)

    def forward(self, input):
        quant_t, quant_b, z_q_x_b, z_q_x_t, enc_b, enc_t = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, (enc_b, z_q_x_b), (enc_t, z_q_x_t)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        enc_t = self.quantize_conv_t(enc_t) #.permute(0, 2, 3, 1)
        quant_t, z_q_x_t = self.quantize_t.straight_through(enc_t)
        #quant_t = quant_t.permute(0, 3, 1, 2)
        #diff_t = diff_t.unsqueeze(0)

        dec_t = self.dec_t(quant_t)
        enc_b = torch.cat([dec_t, enc_b], 1)

        enc_b = self.quantize_conv_b(enc_b) #.permute(0, 2, 3, 1)
        quant_b, z_q_x_b = self.quantize_b.straight_through(enc_b)
        #quant_b = quant_b.permute(0, 3, 1, 2)
        #diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b, z_q_x_b, z_q_x_t, enc_b, enc_t #, id_t, id_b

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)

        return dec
    '''
    def decode_code(self, code_t, code_b):
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
        pre_z_e_x = self.encoder(x)
        z_e_x = self.quantize_conv(pre_z_e_x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        return x_tilde, z_e_x, z_q_x
    '''
class VQVAEMonster1D(nn.Module):
    def __init__(self, in_channel=1, channel=128, n_res_block=2, n_res_channel=32, embed_dim=64, n_embed=32, decay=0.99, #n_embed default 512
                 first_stride=4, second_stride=2):
        super().__init__()

        self.enc_b = Encoder1D(in_channel, channel, n_res_block, n_res_channel, stride=first_stride)
        self.enc_t = Encoder1D(channel, channel, n_res_block, n_res_channel, stride=second_stride)
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)
        self.quantize_t = VQEmbedding(n_embed, embed_dim) #Quantize(embed_dim, n_embed)
        self.dec_t = Decoder1D(embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=second_stride, apply_sigmoid = False)
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)
        self.quantize_b = VQEmbedding(n_embed, embed_dim) #Quantize(embed_dim, n_embed)
        self.upsample_t = Decoder1D(embed_dim, embed_dim, embed_dim, n_res_block, n_res_channel, stride=second_stride)
        self.dec = Decoder1D(embed_dim + embed_dim, in_channel, channel, n_res_block, n_res_channel, stride=first_stride)

    def forward(self, input):
        quant_t, quant_b, z_q_x_b, z_q_x_t, enc_b, enc_t = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, (enc_b, z_q_x_b), (enc_t, z_q_x_t)

    def encode(self, input):
        enc_b = self.enc_b(input)
        enc_t = self.enc_t(enc_b)

        enc_t = self.quantize_conv_t(enc_t) #.permute(0, 2, 3, 1)
        quant_t, z_q_x_t = self.quantize_t.straight_through(enc_t)
        #quant_t = quant_t.permute(0, 3, 1, 2)
        #diff_t = diff_t.unsqueeze(0)

        print(enc_t.shape, 'enc_t')
        print(quant_t.shape, 'quant_T')
        dec_t = self.dec_t(quant_t)
        print(enc_b.shape, 'enc_b')
        print(dec_t.shape, 'dec_t')
        #_,_,h,w = enc_b.shape
        #dec_t = F.interpolate(dec_t, (h,w))
        enc_b = torch.cat([dec_t, enc_b], 1)

        enc_b = self.quantize_conv_b(enc_b) #.permute(0, 2, 3, 1)
        quant_b, z_q_x_b = self.quantize_b.straight_through(enc_b)
        #quant_b = quant_b.permute(0, 3, 1, 2)
        #diff_b = diff_b.unsqueeze(0)
        return quant_t, quant_b, z_q_x_b, z_q_x_t, enc_b, enc_t #, id_t, id_b

    def decode(self, quant_t, quant_b):
        print('quant_t', quant_t.shape)
        upsample_t = self.upsample_t(quant_t)
        print('upsamle_t', upsample_t.shape)
        #_,_,h,w = quant_b.shape
        print('cat upsample_, quant_b', quant_b.shape)
        #upsample_t = F.interpolate(upsample_t,(h,w))
        quant = torch.cat([upsample_t, quant_b], 1)
        dec = self.dec(quant)
        print('dec', dec.shape)

        return dec

class Encoder1D(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 16:
            blocks = [
                nn.Conv2d(in_channel, channel // 8, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//8),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel // 4, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (3,1), padding=(1,0)),
                #nn.ReLU(inplace=True),
            ]

        elif stride == 8:
            blocks = [
                nn.Conv2d(in_channel, channel // 4, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel // 2, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, (4,1) , stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (3,1), padding=(1,0)),
            ]

        elif stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, (1,4), stride=2, padding=(0,1)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, (1,4), stride=2, padding=(0,1)),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, (1,3), padding=(0,1)),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, (1,4), stride=2, padding=(0,1)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, (1,4), padding=(0,1)),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock1D(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder1D(nn.Module):
    def __init__(self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride, apply_sigmoid = True):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, (3,1), padding=(1,0))]

        for i in range(n_res_block):
            blocks.append(ResBlock1D(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 16:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, (4,1), stride=2, output_padding=(1,0)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, channel // 4, (4,1), stride=2, output_padding=(1,0)),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 4, channel // 8, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 8, out_channel, (4,1), stride=2, padding=(1,0)),
            ])

        elif stride == 8:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, channel // 4, (4,1), stride=2, padding=(1,0)),
                nn.BatchNorm2d(channel//4),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 4, out_channel, (4,1), stride=2, padding=(1,0)),
            ])

        elif stride == 4:
            blocks.extend([
                nn.ConvTranspose2d(channel, channel // 2, (1,4), stride=2, padding=(0,1)),
                nn.BatchNorm2d(channel//2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channel // 2, out_channel, (1,4), stride=2, padding=(0,1)),
            ])

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, (4,1), stride=2, padding=(1,0), output_padding=(1,0))
            )

        self.blocks = nn.Sequential(*blocks)
        if apply_sigmoid:
            self.last_layer = nn.Sigmoid()
        else:
            self.last_layer = nn.Identity()

    def forward(self, input):
        #print('----------------DECODER----------------')
        for b in self.blocks:
            input = b(input)
            #print(input.shape)
        input = self.last_layer(input)
        #print('--------------decod fin ---------------')
        return input #self.last_layer(self.blocks(input))



class ResBlock1D(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, (1,3), padding=(0,1)),
            nn.BatchNorm2d(channel), #MY
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
            nn.BatchNorm2d(in_channel), #MY
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out
