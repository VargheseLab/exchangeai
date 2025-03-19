import copy
from torch import zeros, ones
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from torch.nn import Module

class TorchModifier():
    def __init__(
            self,
            old_dim: int,
            new_dim: int
        ) -> None:
        self.old_dim = old_dim
        self.new_dim = new_dim
    
    def modify_torch(self, torch_model: Module):
        
        modified_nodes = []
        self.modified_torch_model = copy.deepcopy(torch_model)
        for name, module in self.modified_torch_model.named_modules():
            for subname, submodule in list(module.named_children()):
                mod_node = self._modify_torch(submodule)
                if mod_node:
                    if len(name) > 0:
                        modified_nodes.append(f"{name}.{subname}")
                    else:
                        modified_nodes.append(subname)

        return self.modified_torch_model, modified_nodes

    def _modify_torch(self, module):
        modified = False
        if hasattr(module, 'out_features') and module.out_features == self.old_dim:
            modified = self._modify_parameters(module, 'out_features', self.new_dim)
        elif hasattr(module, 'out_channels') and module.out_channels == self.old_dim:
            modified = self._modify_parameters(module, 'out_channels', self.new_dim)
        elif hasattr(module, 'num_features') and module.num_features == self.old_dim:
            modified = self._modify_parameters(module, 'num_features', self.new_dim)
        return modified

    

    def _modify_parameters(self, module, attr_name, new_dim):
        old_dim = getattr(module, attr_name)
        setattr(module, attr_name, new_dim)
        
        for name, param in module.named_parameters(recurse=False):
            ndim = list(param.size())
            if old_dim in ndim:
                dim_index = ndim.index(old_dim)
                ndim[dim_index] = new_dim
                new_param_data = zeros(ndim)
                
                if 'weight' in name:
                    if  attr_name == 'num_features':
                        # BatchNorm
                        new_param_data = ones(ndim)
                    else:
                        new_param_data = kaiming_uniform_(new_param_data)
                elif 'bias' in name:
                    new_param_data = zeros(ndim)
                
                param.data = new_param_data

        for buffer_name in ['running_mean', 'running_var']:
            if hasattr(module, buffer_name):
                buffer = getattr(module, buffer_name)
                if buffer is not None and len(buffer) == old_dim:
                    new_buffer = zeros(new_dim)
                    setattr(module, buffer_name, new_buffer)

        return True

