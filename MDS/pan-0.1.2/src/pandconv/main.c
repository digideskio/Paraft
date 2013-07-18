/*******************************************************************************
 *
 * Copyright (C) 2011 Paulo Joia Filho
 * University of Sao Paulo - Sao Carlos/SP, Brazil.
 * All Rights Reserved.
 *
 * This file is part of Projection Analyzer (PAn).
 *
 * PAn is free software: you can redistribute it and/or modify it under
 * the terms of the GNU Lesser General Public License as published by the Free
 * Software Foundation, either version 3 of the License, or (at your option)
 * any later version.
 *
 * PAn is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public 
 * License for more details.
 *
 * This code was developed by Paulo Joia Filho <pjoia@icmc.usp.br>
 * at Institute of Mathematics and Computational Sciences - ICMC
 * University of Sao Paulo, Sao Carlos/SP, Brazil (http://www.icmc.usp.br)
 *
 * Contributor(s):  Luis Gustavo Nonato <gnonato@icmc.usp.br>
 *
 * You should have received a copy of the GNU Lesser General Public License 
 * along with PAn. If not, see <http://www.gnu.org/licenses/>.
 *
 ******************************************************************************/
#include "pan_dconv.h"

#define COPYRIGHT  "|********************************************|\n"   \
                   "| Copyright: PAnDConv 1.0.0  - Apr 2011      |\n"   \
                   "| Projection Analyzer - powered by ANSI C 99 |\n"   \
                   "| Author: Paulo Joia Filho                   |\n"   \
                   "**********************************************\n"

int main(int argc, char *argv[])
{
    /* Parameters */
    //argc=3;
    //argv[1]="pex2weka";
    //argv[2]="../data/sampledata.data";
    //argv[3]=NULL;

    //argv[1]="weka2pex";
    //argv[2]="../data/highdimdata.arff";
    //argv[3]=NULL;

    if (argc < 3)
    {
        textfile_print("pandconv.readme");
        hzline_print(80);
        return EXIT_FAILURE;
    }

    char *function_lwr;
    function_lwr = strdup(argv[1]);
    str2lwr(function_lwr, argv[1]);

    if (strcmp(function_lwr,"pex2weka")==0)
        pex2weka(argv[2], argv[3]);
    else if (strcmp(function_lwr,"weka2pex")==0)
        weka2pex(argv[2], argv[3]);

    return EXIT_SUCCESS;
}


/********************************************* CMD LINE *********************************************/

int pex2weka(char *fullfilename, char *newfilename)
{
    // Import PEx header
    char dstype[maxfld_sz_get()]; // or relation
    char labels[maxfld_sz_get() * maxflds_get()];
    int numrows = 0;
    int numcols = 0;
    unsigned char isconsist = 0;

    pex_importheader(fullfilename, dstype, labels, &numrows, &numcols, &isconsist);
    if (!isconsist)
    {
        printf("\nPEx file %s is inconsistent.\n"
               "Id (first column) and/or Class (last column) were not found.\n"
               "Aborted operation!\n\n", fullfilename);
        return EXIT_FAILURE;
    }

    // Import PEx data
    decimal *dataset = (decimal*) malloc (numrows * numcols * sizeof(decimal));

    struct id_struct ids = {DEFAULT_VALUE};
      ids.values = (char**) malloc (numrows * sizeof(char*));

    struct class_struct classes = {DEFAULT_VALUE};
      classes.values = (char**) malloc (numrows * sizeof(char*));
      classes.enumeration = (char**) calloc (maxclass_get(), sizeof(char*));

    pex_importdata(fullfilename, numrows, numcols, dataset, &ids, &classes);

    // Export to Weka Format
    if (!(newfilename))
        newfilename = strrep(fullfilename, ".data", ".arff");

    pex_labelsconsist(labels, numcols);
    char **attributes = (char**) malloc (numcols * sizeof(char*));
    parse(strdup(labels), ";", NULL, attributes, NULL_INT, NULL_INT);

    weka_export(newfilename, dstype, attributes, numrows, numcols, dataset, ids.values, &classes);

    /* Conclusion and copyright */
    printf("Weka file generated successfully in:\n%s\n\n", newfilename);
    printf(COPYRIGHT);

    // release memory
    free(ids.values);
    free(classes.values);
    free(classes.enumeration);
    free(attributes);
    free(dataset);

    return EXIT_SUCCESS;
}

int weka2pex(char *fullfilename, char *newfilename)
{
    // Import Weka header
    char relation[BUFSIZ]; // <-- must be a high value!
    char **attributes = (char**) malloc (maxflds_get()*sizeof(char*));
    int numrows=0;
    int numcols=0;
    int startrow=0;
    unsigned char hasid=0;
    unsigned char hascl=0;

    weka_importheader(fullfilename, relation, attributes, &numrows, &numcols, &startrow, &hasid, &hascl);

    // Import Weka data
    decimal *dataset = (decimal*) malloc (numrows*numcols*sizeof(decimal));

    struct id_struct ids = {DEFAULT_VALUE};
    if (hasid) {
        ids.retrieve = GET_ID;
        ids.values = (char**) malloc (numrows * sizeof(char*));
    }
    struct class_struct classes = {DEFAULT_VALUE};
    if (hascl) {
      classes.retrieve = GET_CLASS;
      classes.values = (char**) malloc (numrows*sizeof(char*));
      classes.enumeration = (char**) calloc (maxclass_get(), sizeof(char*));
      classes.pex_consist = 1;
    }

    weka_importdata(fullfilename, numrows, numcols, startrow, dataset, &ids, &classes);

    // Write to PEx format
    if (!(newfilename))
        newfilename = strrep(fullfilename, ".arff", ".data");

    char labels[numcols * maxfld_sz_get()];
    strjoin(labels, attributes, ";");

    char *dstype = strdup(relation);
    str2lwr(dstype, relation);
    if ((strcmp(dstype, "dy")) && (strcmp(dstype, "sy")))
        strcpy(relation, "DY");

    pex_export(newfilename, relation, labels, numrows, numcols, dataset, ids.values, &classes);

    /* Conclusion and copyright */
    printf("PEx file generated successfully in:\n%s\n\n", newfilename);
    printf(COPYRIGHT);

    // release memory
    free(attributes);
    free(dataset);
    if (hasid) free(ids.values);
    if (hascl) {free(classes.values); free(classes.enumeration);}
    free(dstype);

    return EXIT_SUCCESS;
}
