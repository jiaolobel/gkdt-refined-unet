
class EvalConfig:
    def __init__(self):
        theta_alphas = [80.0, 40.0, 10.0, 120.0] # 0~3
        theta_betas = [0.03125, 0.0625, 0.125, 0.25] # 0~3
        theta_gamma = 3.0
        ns_processes = [8, 4, 2, 1] # 0~2
        channels = ["rgb", "multiband"] # 0~1

        # Params in terms of vars.
        self.tasks = [
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[0], theta_gamma), 

            # theta_alpha
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[1], theta_betas[0], theta_gamma), 
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[2], theta_betas[0], theta_gamma), 
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[3], theta_betas[0], theta_gamma), 

            # theta_beta
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[1], theta_gamma), 
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[2], theta_gamma), 
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[0], theta_alphas[0], theta_betas[3], theta_gamma), 

            # n_processes
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[1], theta_alphas[0], theta_betas[0], theta_gamma), 
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[0], ns_processes[2], theta_alphas[0], theta_betas[0], theta_gamma), 

            # channels
            "{}/n_processes={}/a={}, b={}, r={}".format(channels[1], ns_processes[0], theta_alphas[0], theta_betas[0], theta_gamma), 
        ] 