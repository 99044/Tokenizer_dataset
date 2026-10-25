import json, torch

def install():
    import sys
    train_ddp=sys.modules['__main__']
    original_build=train_ddp.build_optimizer
    original_clip=torch.nn.utils.clip_grad_norm_
    names={};steps=[0]
    def build(model,*args,**kwargs):
        names.update({id(p):name for name,p in model.named_parameters()})
        bad=[dict(name=n,values=p.detach().cpu().tolist()) for n,p in model.named_parameters() if not torch.isfinite(p).all().item()]
        print(json.dumps(dict(diagnostic='initial_nonfinite_parameters',bad=bad)),flush=True)
        if bad:raise RuntimeError('Nonfinite parameters exist immediately after initialization')
        for name,module in model.named_modules():
            if type(module).__name__ in {'GatedDeltaNet','FeedForward','Attention','LMHead'}:
                def hook(mod,inputs,output,name=name):
                    if steps[0]<26:return
                    if isinstance(output,torch.Tensor) and not torch.isfinite(output).all().item():
                        print(json.dumps(dict(diagnostic='nonfinite_forward',rank=__import__('os').environ.get('RANK'),module=name,type=type(mod).__name__)),flush=True)
                module.register_forward_hook(hook)
        return original_build(model,*args,**kwargs)
    def clip(parameters,*args,**kwargs):
        params=list(parameters);steps[0]+=1
        if steps[0]<26:return original_clip(params,*args,**kwargs)
        grads=[]
        for p in params:
            if p.grad is None:continue
            g=p.grad.detach();finite=bool(torch.isfinite(g).all().item())
            grads.append(dict(name=names[id(p)],finite=finite,max_abs=float(g.abs().max()),norm64=float(torch.linalg.vector_norm(g.double()))))
        bad=[g for g in grads if not g['finite']]
        print(json.dumps(dict(diagnostic='gradient_check',rank=__import__('os').environ.get('RANK'),step=steps[0],nonfinite_count=len(bad),nonfinite=bad[:30],largest=sorted((g for g in grads if g['finite']),key=lambda x:x['max_abs'],reverse=True)[:8])),flush=True)
        return original_clip(params,*args,**kwargs)
    train_ddp.build_optimizer=build
    torch.nn.utils.clip_grad_norm_=clip
