import torch
import numpy as np
from torch.autograd.function import once_differentiable
import torch.nn.functional as F
import sys
import os

import dcls_1d, dcls_2d, dcls_3d, dcls_2_1d, dcls_3_1d, dcls_3_2d, im2col_dcls, dcls

class F_im2col_dcls(torch.autograd.Function):
    @staticmethod
    def forward(ctx, im, P1, P2, dilation, padding, stride, out_dim, shifts):
        ctx.dilation = dilation 
        ctx.padding = padding 
        ctx.stride = stride
        ctx.out_dim = out_dim
        ctx.shifts = shifts         
        
        output = im2col_dcls.forward(im,
                                         P1, P2,
                                         ctx.dilation[0], ctx.dilation[1],
                                         ctx.padding[0], ctx.padding[1],
                                         ctx.stride[0], ctx.stride[1],
                                         ctx.out_dim[0], ctx.out_dim[1],                                     
                                         ctx.shifts[0], ctx.shifts[1]
                                         )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None, None, None, None, None
    

class SurrogateDilation(torch.autograd.Function):
    pass
    
class SurrogateDilation1d(SurrogateDilation):
    
    @staticmethod 
    def forward(ctx, weight, P, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P)
        
        output = dcls_1d.forward(weight,
                                       P,
                                       ctx.dilation[0]
                                      )

        return output


    @staticmethod  
    def backward(ctx, grad_output):
                  
        weight, P = ctx.saved_tensors

        outputs = dcls_1d.backward(weight, 
                                         P, 
                                         grad_output.contiguous(),
                                         ctx.dilation[0]
                                        )
        
        grad_weight, grad_P = outputs


        return grad_weight, grad_P, None
    
class SurrogateDilation2d(SurrogateDilation):
    
    @staticmethod 
    def forward(ctx, weight, P1, P2, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P1, P2)
        
        output = dcls_2d.forward(weight,
                                       P1, 
                                       P2, 
                                       ctx.dilation[0], ctx.dilation[1]
                                      )

        return output


    @staticmethod   
    def backward(ctx, grad_output):
                  
        weight, P1, P2 = ctx.saved_tensors

        outputs = dcls_2d.backward(weight, 
                                         P1, 
                                         P2, 
                                         grad_output.contiguous(),
                                         ctx.dilation[0], ctx.dilation[1]
                                   )
        
        grad_weight, grad_P1, grad_P2 = outputs


        return grad_weight, grad_P1, grad_P2, None

class SurrogateDilation3d(SurrogateDilation):
    
    @staticmethod 
    def forward(ctx, weight, P1, P2, P3, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P1, P2, P3)
        
        output = dcls_3d.forward(weight,
                                       P1, 
                                       P2,
                                       P3,
                                       ctx.dilation[0], ctx.dilation[1], ctx.dilation[2]
                                      )

        return output


    @staticmethod   
    def backward(ctx, grad_output):
                  
        weight, P1, P2, P3 = ctx.saved_tensors

        outputs = dcls_3d.backward(weight, 
                                         P1, 
                                         P2,
                                         P3,
                                         grad_output.contiguous(),
                                         ctx.dilation[0], ctx.dilation[1], ctx.dilation[2]
                                   )
        
        grad_weight, grad_P1, grad_P2, grad_P3 = outputs


        return grad_weight, grad_P1, grad_P2, grad_P3, None

class SurrogateDilation2_1d(SurrogateDilation):
    
    @staticmethod 
    def forward(ctx, weight, P, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P)
        
        output = dcls_2_1d.forward(weight,
                                       P,
                                       ctx.dilation[0]
                                      )
        return output

    @staticmethod  
    def backward(ctx, grad_output):
                  
        weight, P = ctx.saved_tensors

        outputs = dcls_2_1d.backward(weight, 
                                         P, 
                                         grad_output.contiguous(),
                                         ctx.dilation[0]
                                        )
        
        grad_weight, grad_P = outputs

        return grad_weight, grad_P, None 
    
class SurrogateDilation3_1d(SurrogateDilation):
    
    @staticmethod 
    def forward(ctx, weight, P, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P)
        
        output = dcls_3_1d.forward(weight,
                                       P,
                                       ctx.dilation[0]
                                      )
        return output

    @staticmethod  
    def backward(ctx, grad_output):
                  
        weight, P = ctx.saved_tensors

        outputs = dcls_3_1d.backward(weight, 
                                         P, 
                                         grad_output.contiguous(),
                                         ctx.dilation[0]
                                        )
        
        grad_weight, grad_P = outputs

        return grad_weight, grad_P, None

class SurrogateDilation3_2d(SurrogateDilation):
    
    @staticmethod 
    def forward(ctx, weight, P1, P2, dilation):
        
        ctx.dilation = dilation        
        
        ctx.save_for_backward(weight, P1, P2)
        
        output = dcls_3_2d.forward(weight,
                                       P1, 
                                       P2, 
                                       ctx.dilation[0], ctx.dilation[1]
                                      )

        return output


    @staticmethod   
    def backward(ctx, grad_output):
                  
        weight, P1, P2 = ctx.saved_tensors

        outputs = dcls_3_2d.backward(weight, 
                                         P1, 
                                         P2, 
                                         grad_output.contiguous(),
                                         ctx.dilation[0], ctx.dilation[1]
                                   )
        
        grad_weight, grad_P1, grad_P2 = outputs


        return grad_weight, grad_P1, grad_P2, None
    
class RSurrogateDilation(torch.autograd.Function):
    pass

class RSurrogateDilation1d(RSurrogateDilation):

    @staticmethod 
    def forward(ctx, input, weight, bias, stride, padding, dilation_max, groups, dilation):
        
        ctx.stride, ctx.padding, ctx.dilation_max, ctx.groups = stride, padding, dilation_max, groups        
        
        ctx.save_for_backward(input, weight, bias, dilation)
        
        dil = (dilation * dilation_max[0]).int().clamp(1,dilation_max[0]).item()
        pad = padding[0] - (weight.size(2) * (dilation_max[0] - dil)) // 2
        
        output = F.conv1d(input, weight, bias, stride, pad, dil, groups)

        return output


    @staticmethod  
    def backward(ctx, grad_output):
                  
        input, weight, bias, dilation = ctx.saved_tensors
        
        grad_input, grad_weight, grad_bias, grad_dilation = torch.zeros_like(input), torch.zeros_like(weight), torch.zeros_like(bias), torch.zeros_like(dilation) #WIP
        print(weight.size())
        print(grad_output.size())
        return grad_input, grad_weight, grad_bias, None, None, None, None, grad_weight.sum().unsqueeze(0)#WIP
    
    
class dcls_conv(torch.autograd.Function):

    @staticmethod 
    def forward(ctx, input, weight, P1, P2, bias, stride, padding, dilation, groups, chunk_size):
        
        ctx.stride = stride 
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.chunk_size = chunk_size        
        
        ctx.save_for_backward(input, weight, P1, P2, bias)
        
        output = dcls.forward( input,
                                 weight, 
                                 P1, 
                                 P2, 
                                 bias,
                                 ctx.dilation[0], ctx.dilation[1],
                                 ctx.stride[0], ctx.stride[1],
                                 ctx.padding[0], ctx.padding[1],
                                 ctx.groups,
                                 ctx.chunk_size
                                )

        return output


    @staticmethod   
    def backward(ctx, grad_output):
                  
        input, weight, P1, P2, bias = ctx.saved_tensors
        outputs = dcls.backward( input,
                                 weight, 
                                 P1, 
                                 P2, 
                                 grad_output.contiguous(),
                                 bias,
                                 ctx.dilation[0], ctx.dilation[1],
                                 ctx.stride[0], ctx.stride[1],
                                 ctx.padding[0], ctx.padding[1],
                                 ctx.groups,
                                 ctx.chunk_size
                                )
        
        grad_input, grad_weight, grad_P1, grad_P2, grad_bias = outputs


        return grad_input, grad_weight, grad_P1, grad_P2, grad_bias, None, None, None, None, None   