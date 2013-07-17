#include "pan_dconv.h"

int main()
{
    /* data conversion */

    /* PEx -> Weka */
    printf("\n");
    char *fullfilename = "../../data/ssurface.data";
    pex2weka(fullfilename, NULL);

    /* Weka -> PEx */
    printf("\n");
    fullfilename = "../../data/diabetes.arff";
    weka2pex(fullfilename, NULL);

    printf("\n");
    return 0;
}

