This directory contains a model to train a LSTM to deal with various classes of multirotors:

In the environment definition file the data for each class of multirotors are loaded from different files

and stored in the class definition during the reset function execution. For now the classes of multirotor simulated
are from 6 inches to 20 and different parameters are assigned to each class.

To train the agent the linear model is kept becouse it shows better learning performance if the coefficients are 
more or less accurate, then the trained policy can be deployed on the non linear model.

The problem to be solved stay the same of ceating a waypoint navigation.

The files are in the /buildParamsFiles/ and each file contains the parameters for the specific class
so one file for each prop diameter. The name format is standard D_<prop_diam>_in.txt EG.: D_7_in.txt where
"D" stands for Diameter, the number is te prop diameter in inches to which are associated all the other build parameters.

For sake of simplicity all the buids are quadcopter +.

Now total mass of quadrotor is calculated from the mass of various components, this parameter is give by the user agent
at creation, the dimension of the central electronics and the distance between battery and CG stay the same.

The Maximum RPM for the motor + prop are given for each diameter configuration as safety values; this value is used
to evaluate the trim throttle to map actions into commands. this factor have to be taken into account in case of 
real policy deployment.

The value are stored in parameters arrays created in the __init__() method and then the component of the arrays
used is changed at every reset. so the files are opened only at the initialization and the are not used anymore 
since the building parameters are stored in appropriate arrays wit name: <parameter_name>_a, to better separate 
parameters, the initialization variables' names are suffixed with "_i"
