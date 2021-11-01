import os
import json

class Options():
    def get_default():
        opt = {}
        # Input info
        opt["n_dims"]                  = 3              # 2 or 3
        opt['activation_function']     = "sine"         # sine or relu
        opt['use_positional_encoding'] = False
        opt['num_positional_encoding_terms'] = 6
        opt['interpolate']             = False
        opt["vector_field_name"]       = "isotropic1024coarse_ts1.h5"
        opt["save_name"]               = "Implicit_isotropic1024coarse_ts1"    # Folder that the model will be saved to
        opt["n_layers"]                = 8        # Smallest a dimension can go to upscale from
        opt["nodes_per_layer"]         = 1024

        opt["train_distributed"]       = False
        opt["device"]                  = "cuda:0"
        opt['data_device']             = "cuda:0"
        opt["gpus_per_node"]           = 8
        opt["num_nodes"]               = 1
        opt["ranking"]                 = 0

        opt["iterations"]              = 10000
        opt["points_per_iteration"]    = 10000     # Minibatch for training
        opt["lr"]                      = 1e-5   # Learning rate for GAN discriminator
        opt["beta_1"]                  = 0.9
        opt["beta_2"]                  = 0.999

        opt["iteration_number"]        = 0
        opt["save_every"]              = 100
        opt['log_every']               = 5

        return opt

def save_options(opt, save_location):
    with open(os.path.join(save_location, "options.json"), 'w') as fp:
        json.dump(opt, fp, sort_keys=True, indent=4)
    
def load_options(load_location):
    opt = Options.get_default()
    print(load_location)
    if not os.path.exists(load_location):
        print("%s doesn't exist, load failed" % load_location)
        return
        
    if os.path.exists(os.path.join(load_location, "options.json")):
        with open(os.path.join(load_location, "options.json"), 'r') as fp:
            opt2 = json.load(fp)
    else:
        print("%s doesn't exist, load failed" % "options.json")
        return
    
    # For forward compatibility with new attributes in the options file
    for attr in opt2.keys():
        opt[attr] = opt2[attr]

    return opt
