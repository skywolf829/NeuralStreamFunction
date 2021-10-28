import sys
import os.path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import numpy as np
import matplotlib.pyplot as plt
import argparse
from utility_functions import load_obj
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a trained SSR model')
    
    parser.add_argument('--save_folder',default="Isomag3D_vis_results",
        type=str,help='Folder to save images to')
    parser.add_argument('--output_file_name',default="Isomag3D.results",
        type=str,help='filename to visualize in output folder')
    parser.add_argument('--mode',type=str,default="3D")
    parser.add_argument('--start_ts', default=4000, type=int)
    parser.add_argument('--ts_skip', default=100, type=int)
    
    

    #plt.style.use('Solarize_Light2')
    #plt.style.use('fivethirtyeight')
    plt.style.use('ggplot')
    #plt.style.use('seaborn')
    #plt.style.use('seaborn-paper')

    font = {#'font.family' : 'normal',
        #'font.weight' : 'bold',
        'font.size'   : 22,
        'lines.linewidth' : 5}
    plt.rcParams.update(font)

    args = vars(parser.parse_args())

    project_folder_path = os.path.dirname(os.path.abspath(__file__))
    project_folder_path = os.path.join(project_folder_path, "..", "..")
    data_folder = os.path.join(project_folder_path, "Data", "SuperResolutionData")
    output_folder = os.path.join(project_folder_path, "Output")
    save_folder = os.path.join(project_folder_path, "SavedModels")
    results_file = os.path.join(output_folder, args['output_file_name'])
    
    results = load_obj(results_file)
    save_folder = os.path.join(output_folder, args['save_folder'])
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    for scale_factor in results.keys():
        if not os.path.exists(os.path.join(save_folder, scale_factor)):
            os.makedirs(os.path.join(save_folder, scale_factor))
    if not os.path.exists(os.path.join(save_folder, "MedianValues")):
        os.makedirs(os.path.join(save_folder, "MedianValues"))
    

    
    interp = "bilinear" if args['mode'] == "2D" else "trilinear"

    for scale_factor in results.keys():
        print(scale_factor)

        interp_results = results[scale_factor][interp]


        for metric in interp_results.keys():
            fig = plt.figure()
            y_label = metric

            for SR_type in results[scale_factor].keys():

                # model results plotting
                x = np.arange(args['start_ts'], 
                    args['start_ts'] + args['ts_skip']*len(results[scale_factor][SR_type][metric]),
                    args['ts_skip'])
                y = results[scale_factor][SR_type][metric]
                l = SR_type
                if(SR_type == "ESRGAN" or SR_type == "SSRTVD"):
                    l = SR_type + " hierarchy"
                plt.plot(x, y, label=l)

            plt.legend()
            plt.xlabel("Timestep")
            plt.ylabel(y_label)

            plt.title(scale_factor + " SR - " + metric)
            plt.savefig(os.path.join(save_folder, scale_factor, metric+".png"),bbox_inches='tight',dpi=100)
            plt.clf()

    # Overall graphs

    averaged_results = {}

    scale_factors = []

    for scale_factor in results.keys():

        scale_factor_int = int(scale_factor.split('x')[0])
        scale_factors.append(scale_factor_int)

        for metric in results[scale_factor][interp].keys():
            for SR_type in results[scale_factor].keys():
                if SR_type not in averaged_results.keys():
                    averaged_results[SR_type] = {}

                if(metric not in averaged_results[SR_type].keys()):
                    averaged_results[SR_type][metric] = []

                averaged_results[SR_type][metric].append(np.median(
                    np.array(results[scale_factor][SR_type][metric])))

    
    for metric in averaged_results[interp].keys():
        fig = plt.figure()
        y_label = metric
        for SR_type in averaged_results.keys():
            # model results plotting
            x = scale_factors
            y = averaged_results[SR_type][metric]
            l = SR_type
            if(SR_type == "ESRGAN" or SR_type == "SSRTVD"):
                l = SR_type + " hierarchy"
            plt.plot(x, y, label=l)

        #plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
        #        mode="expand", borderaxespad=0, ncol=3)
        if(metric == "SSIM"):
            plt.xlabel("Scale factor")
        if(args['output_file_name'] == "Isomag2D.results"):
            plt.ylabel(y_label)
        plt.xscale('log')
        plt.minorticks_off()
        plt.xticks(scale_factors, labels=scale_factors)
        #plt.title("Median " + metric + " over SR factors")
        if(metric == "PSNR (dB)"):
            plt.ylim(bottom=20, top=55)
            plt.title(args['output_file_name'].split(".")[0])
        elif(metric == "SSIM"):
            plt.ylim(bottom=0.45, top=1.0)
        plt.savefig(os.path.join(save_folder, "MedianValues", metric+".png"),
            bbox_inches='tight',
            dpi=100)
        #plt.show()
        plt.clf()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    left_y_label = "PSNR (dB)"
    right_y_label = "SSIM"

    for SR_type in averaged_results.keys():
        # model results plotting
        x = scale_factors
        left_y = averaged_results[SR_type][left_y_label]
        right_y = averaged_results[SR_type][right_y_label]
        l = SR_type
        if(SR_type == "ESRGAN" or SR_type == "SSRTVD"):
            l = SR_type + " hierarchy"
        ax1.plot(x, left_y, label=l, marker="s")
        ax2.plot(x, right_y, label=l, marker="^", linestyle='dashed')

    ax1.legend()
    #ax2.legend()
    ax1.set_xlabel("Scale factor")
    ax1.set_ylabel(left_y_label)
    ax2.set_ylabel(right_y_label)

    ax1.set_xscale('log')
    ax1.minorticks_off()

    ax1.set_xticks(scale_factors)
    ax1.set_xticklabels(scale_factors)
    ax1.set_title("Median PSNR/SSIM over SR factors")
    plt.savefig(os.path.join(save_folder, "MedianValues", "Combined.png"),bbox_inches='tight',dpi=100)
    plt.clf()