import sys
sys.path.insert(0, './scripts')

import volumesToSlices
import padToSize
import applyNetwork_unet
import applyNetwork_vnet

def main(argv):

    dataset_name = "slices_tellus"

    volume_list_path = "config/volume_list.txt"
    output_path = "output/" + dataset_name + "/"

    # Choose network
    net = "unet"
    #net = "vnet"

    # Convert listed volumes to slices as network input
    volumesToSlices.volumesToSlices(dataset_name, volume_list_path)

    # Apply network
    if net == "unet":
        network_path = "network_snapshots/u_net.pth.tar"
        applyNetwork_unet.processSlices("data/" + dataset_name, network_path, output_path)

    elif net == "vnet":
        padToSize.padToSize("data/" + dataset_name, 24)

        network_path = "network_snapshots/v_net.pth.tar"
        applyNetwork_vnet.processSlices("data/" + dataset_name, network_path, output_path)


if __name__ == '__main__':
    main(sys.argv)
