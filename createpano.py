#!/usr/bin/python
import os
import sys

prefix = sys.argv[1]

os.system("convert -append {0}_dyn_1409.png {0}_dyn_1022.png out1.png".format(prefix))
#os.system("convert -append {0}_2Dacf_1409.png {0}_2Dacf_1022.png out2.png".format(prefix))
#os.system("convert -append {0}_arcfit*.png out3.png".format(prefix))
#os.system("convert -append {0}_1Dacf*.png out4.png".format(prefix))
#os.system("convert -append {0}_scintpars.png {0}_table.png out5.png".format(prefix))
os.system("convert +append {0}_fulldyn.png out*.png {0}_panorama.png".format(prefix))
os.system("rm out*.png")
