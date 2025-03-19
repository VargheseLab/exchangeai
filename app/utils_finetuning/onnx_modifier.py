import numpy as np
import onnx
import onnx.numpy_helper as numpy_helper

class OnnxModifier():
    def __init__(
            self,
            old_dim: int,
            new_dim: int
        ) -> None:
        self.old_dim = old_dim
        self.new_dim = new_dim
        

    def modify_raw_data_onnx(self, raw_data):
        # get old data
        conv_data = numpy_helper.to_array(raw_data)

        if False:
            # keep weights, and add new weight to raw_data with uniform initialized data
            conv_data = np.concatenate([
                    conv_data,
                    np.repeat(
                        np.expand_dims(np.median(conv_data, axis=0), axis=0),
                        self.new_dim - self.old_dim,
                        axis=0
                    )
                ], axis=0)
        else:
            # generate random new numbers
            # generate random unfiform numbers from -1 to 1
            conv_data = (
                (
                    np.random.default_rng().uniform(-1, 1, (self.new_dim, *conv_data.shape[1:]) )
                )
            ).astype(np.float32)
        return conv_data
    
    def modify_onnx(self, onnx_model):
        # Modify the last layer of the model to have an additional output class
        nodes_names = [node.name for node in onnx_model.graph.value_info]
        nodes_names.extend(node.name for node in onnx_model.graph.initializer)
        nodes_names.extend(node.name for node in onnx_model.graph.output)
        nodes_names.extend(node.name for node in onnx_model.graph.input)
        
        modified_nodes = []
        for _ in range(5):
            onnx_model, _modified_nodes = self._modify_onnx(onnx_model, nodes_names)
            if _modified_nodes == []:
                break
            modified_nodes.extend(_modified_nodes)

        # rename to torch convention
        modified_nodes = set(["/".join(m.split('.')[:-1]) for m in modified_nodes])

        return onnx_model, modified_nodes

    def _modify_onnx(self, onnx_model, nodes_names):
        # Adapt initializer nodes and weights
        new_nodes = []
        modified_nodes = []

        for node in onnx_model.graph.initializer:
            if (node.name in nodes_names) and 'batch' not in node.name:
                node_ = node.__deepcopy__()
                dims_ = np.asarray(node_.dims)

                try: 
                    dims = np.nonzero(dims_ == self.old_dim)[0][0]
                except IndexError: pass
                except Exception as e: print(e)
                else:
                    try:
                        # overwrite dimensions
                        dims = [*dims_[:dims], self.new_dim, *dims_[dims+1:]]
                        dims = [d if d != 0 else "s0" for d in dims]

                        conv_data = self.modify_raw_data_onnx(node_)
                        new_nodes.append(onnx.helper.make_tensor(
                                name = node_.name,
                                data_type = node_.data_type,
                                dims = dims,
                                vals = conv_data,
                        ))
                        modified_nodes.append(node.name)
                        onnx_model.graph.initializer.remove(node)
                    except Exception as e: print(e)
            else:
                pass
                #logging.warning(f"{node.name} {node}")

        onnx_model.graph.initializer.extend(new_nodes)

        # Adapt value infos if exported with (keep_initializers_as_inputs=True)
        new_nodes = []
        for node in onnx_model.graph.value_info:
            if (node.name in nodes_names):
                new_node = self.modify_value_info(node)
                if new_node:
                    new_nodes.append(new_node)
                    onnx_model.graph.value_info.remove(node)

        onnx_model.graph.value_info.extend(new_nodes)

        # Adapt value infos if exported with (keep_initializers_as_inputs=True)
        new_nodes = []
        for node in onnx_model.graph.input:
            if (node.name in nodes_names):
                new_node = self.modify_value_info(node)
                if new_node:
                    new_nodes.append(new_node)
                    onnx_model.graph.input.remove(node)

        onnx_model.graph.input.extend(new_nodes)
                        

        # This is for batch normalization layers
        new_nodes = []
        for node in onnx_model.graph.node:
            if 'batch' in node.name.lower():
                for attr in node.attribute:
                    if attr.name == 'training_mode':
                        attr.i = 0
                        continue

        # BatchNorm remove running outputs which are not connected but cause errors
        for node in onnx_model.graph.node:
            if 'batch' in node.name.lower():
                while len(node.output) > 1:
                    node.output.remove(node.output[1])
                for attr in node.attribute:
                    if 'training_mode' == attr.name:
                        attr.i = 0

        # adapt output node
        node = onnx_model.graph.output[0]
        new_node = self.modify_value_info(node)
        if new_node:
            onnx_model.graph.output.remove(node)
            onnx_model.graph.output.append(new_node)
        
        return onnx_model, modified_nodes
    
    def modify_value_info(self, protoTensor):
        node_ = protoTensor.__deepcopy__()
        
        val = node_.type.tensor_type
        dims = val.shape.dim
        
        val_dims = np.asarray([d.dim_value for d in dims])
        try: 
            dims = np.nonzero(val_dims == self.old_dim)[0][0]
        except Exception: 
            return None
        else:
            if isinstance(val_dims, int): dims = self.new_dim
            else: 
                dims = [*val_dims[:dims], self.new_dim, *val_dims[dims+1:]]
                dims = [int(d) if d != 0 else "s0" for d in dims]

            new_node =  onnx.helper.make_tensor_value_info(
                    name = node_.name,
                    elem_type = val.elem_type,
                    shape = dims
                )
            return new_node
            