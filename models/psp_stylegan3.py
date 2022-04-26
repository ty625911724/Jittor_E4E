import jittor as jt
from jittor import nn
from models.encoders import psp_encoders
from models.stylegan3.networks_stylegan3 import Generator
import numpy as np

def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt

class pSp(nn.Module):
    def __init__(self, opts):
        super(pSp, self).__init__()
        self.opts = opts
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(z_dim=512, c_dim=0, w_dim=512,img_resolution=1024, img_channels=3)
        self.face_pool = jt.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights if needed
        self.load_weights()

    def set_encoder(self):
        if self.opts.encoder_type == 'GradualStyleEncoder':
            #encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se', self.opts)
            print("Not implement!!!!!!")
        elif self.opts.encoder_type == 'Encoder4Editing':
            encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
        elif self.opts.encoder_type == 'SingleStyleCodeEncoder':
            print("Not implement!!!!!!")
            #encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se', self.opts)
        else:
            raise Exception('{} is not a valid encoders'.format(self.opts.encoder_type))
        return encoder

    def load_weights(self):
        if self.opts.checkpoint_path is not None:
            print('Loading e4e over the pSp framework from checkpoint: {}'.format(self.opts.checkpoint_path))
            #ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
            ckpt = jt.load(self.opts.checkpoint_path)
            self.encoder.load_state_dict(get_keys(ckpt, 'encoder'))
            self.decoder.load_state_dict(get_keys(ckpt, 'decoder'))
            self.__load_latent_avg(ckpt)
        else:
            print("Don't have weights, This version do not support training")
            #print('Loading encoders weights from irse50!')
            #encoder_ckpt = torch.load(model_paths['ir_se50'])
            #self.encoder.load_state_dict(encoder_ckpt, strict=False)
            #print('Loading decoder weights from pretrained!')
            #ckpt = torch.load(self.opts.stylegan_weights)
            #self.decoder.load_state_dict(ckpt['g_ema'], strict=False)
            #self.__load_latent_avg(ckpt, repeat=self.encoder.style_count)

    def execute(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if self.opts.start_from_latent_avg:
                if codes.ndim == 2:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
                else:
                    print(codes.shape)
                    print(self.latent_avg.repeat(codes.shape[0], 1, 1).shape)
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        #input_is_latent = not input_code
        #images, result_latent = self.decoder([codes],
        #                                     input_is_latent=input_is_latent,
        #                                     randomize_noise=randomize_noise,
        #                                     return_latents=return_latents)
        images = self.decoder.synthesis(codes)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, codes
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            #print(ckpt['latent_avg'])
            #self.latent_avg = jt.array(ckpt['latent_avg'].detach().cpu().numpy())
            self.latent_avg = jt.array(ckpt['latent_avg'])
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None

