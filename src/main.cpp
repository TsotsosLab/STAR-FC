/*
 * Author: yulia
 * Date: 18/11/2016
 */

#include "Controller.hpp"

int main(int argc, char *argv[]) {

    Controller c = Controller();
    c.parseCommandLineOptions(argc, argv);
    c.printParameters();
    c.run();
}
