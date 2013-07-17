#include "pan_dconv.h"
#include "pan_force.h"
#include "pan_lamp.h"
#include "pan_estimate.h"

int lampmethod();

int main()
{
    lampmethod(); // using PEx format !!!
    return 0;
}

int lampmethod()
{
    int numpoints = 0;
    int numsamples = 0;
    int highdim = 0;
    const int projdim = 2;
    char dstype[maxfld_sz_get()]; // PEx: DY=dense dataset; SY= sparse dataset
    char *inputfilename;
    char outputfilename[BUFSIZ];

    /*
    *   Parameters
    */
    inputfilename = "../../data/wdbc-std.data";    //<-- change here for using another dataset!
    strcpy(outputfilename, strrep(inputfilename, ".data","_lamp.prj"));

    // Stress and plot
    boolean stress = False; // <-- for large data sets can increase the processing time considerably!
    boolean plot = False;   // <-- require python packages
    char *legend = "yes";   // <-- just if plot = True
    char *saveim = "yes";   // <-- just if plot = True

    /*
    *   Data reading
    */
    pex_importheader(inputfilename, dstype, NULL, &numpoints, &highdim, NULL);

    decimal *data = (decimal*) malloc (numpoints * highdim * sizeof(decimal));
    struct id_struct ids = {DEFAULT_VALUE};
      ids.values = (char**) malloc (numpoints * sizeof(char*));
    struct class_struct classes = {DEFAULT_VALUE};
      classes.values = (char**) malloc (numpoints * sizeof(char*));

    pex_importdata(inputfilename, numpoints, highdim, data, &ids, &classes);

    /*
    *   Sampling (Control Points)
    */
    numsamples = sqrt(numpoints) * 3;

    struct sampling_struct sampleinfo = {DEFAULT_VALUE};
      sampleinfo.numpoints = numpoints;
      sampleinfo.highdim = highdim;
      sampleinfo.numsamples = numsamples;

    decimal *sampledata = (decimal*) malloc (numsamples * highdim * sizeof(decimal));
    char **sampleid = (char**) malloc (numsamples * sizeof(char*));
    sampling_execute(data, ids.values, &sampleinfo, sampledata, sampleid);

    /*
    *   Force-scheme
    */
    struct idmap_struct idmapinfo = {DEFAULT_VALUE};
      idmapinfo.numpoints = numsamples;
      idmapinfo.highdim = highdim;
      idmapinfo.projdim = projdim;
      idmapinfo.numiterations = 100;

    decimal *sampleproj = (decimal*) malloc (numsamples * projdim * sizeof(decimal));
    idmap_execute(sampledata, sampleid, &idmapinfo, sampleproj);

    /*
    *   Lamp method
    */
    struct lamp_struct lampinfo = {DEFAULT_VALUE};
      lampinfo.numpoints = numpoints;
      lampinfo.numsamples = numsamples;
      lampinfo.highdim = highdim;
      lampinfo.projdim = projdim;

    decimal *projection = (decimal*) malloc (numpoints * projdim * sizeof(decimal));
    lamp_execute(data, sampledata, sampleproj, &lampinfo, projection);

    /*
    *   Stress computation
    */
    decimal stresssamp = 0.0, stressproj = 0.0, stresstime = 0.0;
    struct stress_struct stressinfo = {DEFAULT_VALUE};
    if (stress == True)
    {
        stressinfo.numpoints = numsamples;
        stressinfo.highdim = highdim;
        stressinfo.projdim = projdim;

        // control points
        stresssamp = stress_calc(sampledata, sampleproj, &stressinfo);
        stresstime = stress_elapsedtime();

        // projection
        stressinfo.numpoints = numpoints;
        stressproj = stress_calc(data, projection, &stressinfo);
        stresstime += stress_elapsedtime();
    }

    /*
    *   Data writing
    */
    char *labels = "x;y";
    pex_export(outputfilename, dstype, labels, numpoints, projdim, projection, ids.values, &classes);

    /*
    *   Release memory
    */
    free(data);
    free(ids.values);
    free(classes.values);
    free(sampledata);
    free(sampleid);
    free(sampleproj);
    free(projection);

    /*
    *   Show results
    */
    printf("\n");
    hzline_print(80);
    centerline_print("LAMP Computation", 80);
    hzline_print(80);
    printf("Input dataset:     %s\n", inputfilename);
    printf("Output projection: %s\n", outputfilename);
    printf("Number of points (instances):    %d\n", numpoints);
    printf("Number of samples (ctrl points): %d\n", numsamples);
    printf("High dimension:                  %d\n", highdim);
    printf("Projection dimension:            %d\n", projdim);
    if (stress == True)
    {
        printf("Stress of control points:        %.7f (%s)\n", stresssamp,
               stress_enum_descr(stressinfo.stresstype));
        printf("Stress of projection:            %.7f (%s)\n\n", stressproj,
               stress_enum_descr(stressinfo.stresstype));
        printf("Stress execution time:           %.3f s\n", stresstime);
    }
    printf("LAMP execution time:             %.3f s\n", lamp_elapsedtime());
    hzline_print(80);
    printf("\n");

    /*
    *   Plot results
    */
    if (plot == True)
    {
        char cmdline[BUFSIZ];
        printf("Ploting projection... wait a moment!\n\n");
        sprintf(cmdline, "cd ../../app/python; ./pan_plotproj2d.py -infile "
                "%s -legend %s -saveim %s &", outputfilename, legend, saveim);
        return system(cmdline);
    }

    return EXIT_SUCCESS;
}
