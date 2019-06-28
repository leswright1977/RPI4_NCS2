# RPI4_NCS2
Raspberrry pi 4 OpenVINO Python

The Raspberry pi 4 is now out, and runs the latest version of Raspbian 10 Buster!
Currently if you follow the instruction at: https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html
to install OpenVino, although the builds complete sucessfully for the C++ programs shipped with it, the Python bindings fail.

The package supplied by Intel does not expect Python 3.7, or Rasbian Buster for that matter.
I imagine they will release an update at some point, but I am not very patient.
Here is a workaround, that can get you up and running with OpenVINO on the pi4! 
It appears to work well, at least with my scripts from the PI3. I am getting about 30FPS with the async script on a single NCS2!
That said, there may be things I have not taken into account, or even broken, that I am not aware of!


Initial setup:

Create a directiory in your home folder:

cd ~
mkdir openvino
cd openvino

Install cmake, if you want to build the shipped C++ examples:

sudo apt-get install cmake

Go and grab the lates openvino toolkit (This assumes you are still in ~openvino):

wget https://download.01.org/opencv/2019/openvinotoolkit/l_openvino_toolkit_raspbi_p_2019.1.094.tgz

Untar:

tar -xf l_openvino_toolkit_raspbi_p_2019.1.094.tgz

REPLACE ~/openvino/inference_engine_vpu_arm/bin/setupvars.sh with the version of setupvars.sh that I have uploaded here:
https://github.com/leswright1977/RPI4_NCS2/blob/master/src/setupvars.sh 
About the file:
Fixed expected OS version
Hacked in some symlinks and dynamically loaded libs to fix errors. 

Enable the environment (note we force the script to believe we are running python 3.5)
source ~/openvino/inference_engine_vpu_arm/bin/setupvars.sh -pyver 3.5

Echo the following line into ~./bashrc so the env is loaded at logon:

echo "source ~/openvino/inference_engine_vpu_arm/bin/setupvars.sh -pyver 3.5" >> ~/.bashrc

Add the user 'pi' to the users group:
sudo usermod -a -G users pi

Run the following script:
sh ~/openvino/inference_engine_vpu_arm/install_dependencies/install_NCS_udev_rules.sh

Done!

*Note: if you scripts currently have an import line like this:
from openvino.inference_engine import IENetwork, IEPlugin

As of the latest version of openvino, this is now:
from armv7l.openvino.inference_engine import IENetwork, IEPlugin



