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
#define PRGNAME "PAnDConv"

// Protected member variables
static unsigned short int m_maxflds = 1024;  /* maximum possible number of fields  */
static unsigned char m_maxfld_sz = 80;       /* longest possible field + 1         */
static unsigned short int m_maxclass = 1024; /* maximum possible number of classes */
static decimal *m_dbldataset;

inline unsigned short int maxflds_get()
{
    return m_maxflds;
}
inline void maxflds_set(unsigned short int value)
{
    m_maxflds = value;
}
inline unsigned char maxfld_sz_get()
{
    return m_maxfld_sz;
}
inline void maxfld_sz_set(unsigned char value)
{
    m_maxfld_sz = value;
}
unsigned short int maxclass_get()
{
    return m_maxclass;
}
void maxclass_set(unsigned short int value)
{
    m_maxclass = value;
}

int csv_getdimension(char *fullfilename, char *delim, int *numrows, int *numcols,
                     int *startrow, int *endrow, int *startcol, int *endcol)
{
    /* input validation */
    assert(*startrow < 0 || *endrow < 0);
    assert(*startcol < 0 || *endcol < 0);
    assert(*endrow < *startrow && *endrow);
    assert(*endcol < *startcol && *endcol);

    /* open file in text read mode */
    FILE *infp=fopen(fullfilename,"rb");
    if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

	/* get rows and cols amount */
	*numrows=0;
    *numcols=1;
    char thisline[m_maxflds * m_maxfld_sz];

	while(fgets(thisline,sizeof(thisline),infp)!=0)
	{
	    if (*numrows==*startrow)
            parse(thisline,delim,numcols,NULL,NULL_INT,NULL_INT);
        *numrows+=1;
	}

    /* adjust dimensions */
    assert(*startrow > *numrows || *endrow > *numrows);
    assert(*startcol > *numcols || *endcol > *numcols);
    *numrows = *endrow   > 0 ? *endrow     : *numrows;
    *numrows-= *startrow > 0 ? *startrow-1 : 0;
    *numcols = *endcol > 0   ? *endcol     : *numcols;
    *numcols-= *startcol > 0 ? *startcol-1 : 0;

    if (!*startrow) *startrow = 1;
    if (!*startcol) *startcol = 1;
    if (!*endrow)   *endrow   = *numrows + *startrow - 1;
    if (!*endcol)   *endcol   = *numcols + *startcol-1;
    *startrow-=1; *endrow-=1;
    *startcol-=1; *endcol-=1;

    fclose(infp);
    return EXIT_SUCCESS;
}

int csv_importdata(char *fullfilename, char *delim, int numrows, int numcols,
                   int startrow, int endrow, int startcol, int endcol, char **dataset)
{
    /* open file in text read mode */
    FILE *infp=fopen(fullfilename,"r");
    if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    char thisline[m_maxflds * m_maxfld_sz];
    char **datarow = (char**) malloc (numcols*sizeof(char*));
    int numrow=-1;
    int numcol=0;
    int pos=0;

    /* fill the dataset */
    while(fgets(thisline,sizeof(thisline),infp)!=0)
	{
	    numrow++;

	    if (numrow<startrow)
            continue;
        if (numrow>endrow)
            break;

        parse(thisline,delim,NULL,datarow,startcol,endcol);

        for (numcol=0; numcol < numcols; numcol++)
        {
            pos = (numrow-startrow) * numcols + numcol;
            *(dataset+pos) = (char*) malloc(m_maxfld_sz * sizeof(char));
            strcpy(*(dataset+pos), *(datarow+numcol));
        }
	}

    fclose(infp);
    return EXIT_SUCCESS;
}

int decptr_write(char *fullfilename, decimal *dataset, char *delim, int numrows,
                 int numcols, char insertid, char* textbefore)
{
    char **chrdataset = (char**) malloc (1*sizeof(char*));
    *chrdataset = "USE_DECIMAL_POINTER";
    m_dbldataset = dataset;
    charptr_write(fullfilename, chrdataset, delim, numrows, numcols, insertid, textbefore);
    free(chrdataset);

    return EXIT_SUCCESS;
}

int charptr_write(char *fullfilename, char **dataset, char *delim, int numrows,
                  int numcols, char insertid, char* textbefore)
{
    // open file in text write mode
    FILE *outfp=fopen(fullfilename,"w");
    if(outfp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    char *thisline = (char*) malloc ((numcols+1) * m_maxfld_sz * sizeof(char));
    char strval[m_maxfld_sz];
    int nrow=0;
    int ncol=0;
    int pos=0;

    if (textbefore)
    {
        fprintf(outfp, "%s", textbefore);
    }
    for (nrow=0; nrow < numrows; nrow++)
    {
        strcpy(thisline, "");

        for (ncol=0; ncol < numcols; ncol++)
        {
            pos = nrow * numcols + ncol;
            if ((insertid) && (ncol==0))
                sprintf(thisline, "%d%s", nrow, delim);

            if (strcmp(*dataset,"USE_DECIMAL_POINTER"))
                sprintf(strval, "%s", *(dataset+pos));
            else
                sprintf(strval, "%g", *(m_dbldataset+pos));

            strcat(thisline, strval);

            if (ncol < numcols-1)
                strcat(thisline, delim);
        }
        fprintf(outfp, "%s\n", thisline);
    }

    fclose(outfp);
    free(thisline);

    return EXIT_SUCCESS;
}

int pex_importheader(char *fullfilename, char *dstype, char *labels, int *numrows,
                     int *numcols, unsigned char *isconsist)
{
    FILE *infp=fopen(fullfilename,"r");
    if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    int thisline_sz = m_maxflds * m_maxfld_sz * sizeof(char);
	char *thisline = (char*) malloc(thisline_sz);
	char *thisline_;

	char **datarow = (char**) malloc (m_maxflds * sizeof(char*));
	int fldcnt = 0;

	*numrows = 0;
    *numcols = 0;

    int i=0;
	while(fgets(thisline,thisline_sz,infp)!=0)
	{
	    i++;
        thisline_ = strdup(thisline);
        strrspc(thisline_, thisline);

        switch(i) {
            case 1:
                if(dstype)
                    strcpy(dstype, thisline_);
                break;
            case 2:
                *numrows = atoi(thisline_);
                break;
            case 3:
                *numcols = atoi(thisline_);
                break;
            case 4:
                if(labels)
                    strcpy(labels, thisline_);
                break;
            case 5:
                if (isconsist) {
                    parse(thisline_, ";", &fldcnt, datarow, 0, 0);
                    *isconsist = (fldcnt == *numcols+2) ? 1 : 0;
                }
        }

        free(thisline_);
        if (i >= 5) break;
	}

    fclose(infp);

    // release memory
    free(thisline);
    free(datarow);

    return EXIT_SUCCESS;
}

char* pex_labelsconsist(char *labels, int numcols)
{
    if (!(strcmp(labels, "")))
    {
        pex_labelscreate(labels, numcols);
    }
    else
    {
        size_t sz = strlen(labels);
        if (labels[sz-1] == ';')
            labels[sz-1] = '\0';

        char **attributes = (char**) malloc (m_maxflds*sizeof(char*));
        char *labels_ = strdup(labels);
        int fldcnt = 0;

        parse(labels_, ";", &fldcnt, attributes, 0, 0);

        if (fldcnt != numcols)
            pex_labelscreate(labels, numcols);
        else {
            char jump=0;
            int i, j;
            for (i=0; i<fldcnt-1; i++)
            {
                for (j=i+1; j<fldcnt; j++) {
                    if (strcmp(*(attributes+i), *(attributes+j)) == 0) {
                        pex_labelscreate(labels, numcols);
                        jump = 1;
                        break;
                    }
                }
                if (jump) break;
            }
        }

        free(labels_);
        free(attributes);
    }

    return labels;
}

char* pex_labelscreate(char *labels, int numcols)
{
    strcpy(labels, "");

    int i;
    for(i=1; i <= numcols; i++)
    {
        strcat(labels, FIELD_NAME(i));
        if(i < numcols) strcat(labels, ";");
    }

    return labels;
}

int pex_importdata(char *fullfilename, int numrows, int numcols, decimal *dataset,
                   struct id_struct *ids, struct class_struct *classes)
{
    if (ids->retrieve == DEFAULT_VALUE) ids->retrieve = GET_ID;
    if (ids->size == DEFAULT_VALUE) ids->size = numrows;
    if (classes->retrieve == DEFAULT_VALUE) classes->retrieve = GET_CLASS;
    if (classes->size == DEFAULT_VALUE) classes->size = numrows;

    // exist id and class in file?
	int idcol, clcol;
	idcol = (ids->retrieve == NO_ID_IN_FILE) ? 0 : 1;
	clcol = (classes->retrieve == NO_CLASS_IN_FILE) ? 0 : 1;

	int thisline_sz = m_maxfld_sz * (numcols+idcol+clcol) * sizeof(char);
	char *thisline = (char*) malloc(thisline_sz);
	char **datarow = (char**) malloc((numcols+idcol+clcol) * sizeof(char*));
	int fldcnt = 0;
	int recordcnt = 0;
	char clvalue[m_maxfld_sz]; // current class value
	size_t cllen;              // length of class value

	FILE *infp=fopen(fullfilename,"r");
	if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

	// jump the header
    int i=0;
	while(fgets(thisline,thisline_sz,infp)!=0) {
	    i++;
        if (i >= 4) break;
	}

    // get dataset
	while(fgets(thisline,thisline_sz,infp)!=0)
	{
        parse(thisline,";",&fldcnt,datarow,0,0);

        if (ids->retrieve == GET_ID) {
            *(ids->values + recordcnt) = (char*) malloc((strlen(*datarow)+1)*sizeof(char));
            strcpy(*(ids->values + recordcnt), *datarow);
        }

        for(i=0;i<fldcnt-idcol-clcol;i++)
            *(dataset+i+recordcnt*numcols) = atof(*(datarow+i+idcol));

        if (classes->retrieve == GET_CLASS)
        {
            cllen = strlen(*(datarow+fldcnt-1));
            strrspc(clvalue, *(datarow+fldcnt-1));

            *(classes->values + recordcnt) = (char*) malloc((cllen+1) * sizeof(char));
            strcpy(*(classes->values + recordcnt), clvalue);

            if (classes->enumeration)
            {
                if ((classes->enum_size < m_maxclass) && (classes->enum_size < numrows-1))
                    classes->enum_size = charptr_distinctvaladd(classes->enumeration, clvalue);
                else {
                    classes->enumeration = NULL;
                    classes->enum_size = 0;
                }
            }
        }

        recordcnt++;
	}

    fclose(infp);

    // only one class is not permitted
    if (classes->enum_size == 1) {
        classes->enumeration = NULL;
        classes->enum_size = 0;
    }

    // release memory
    free(datarow);
    free(thisline);

    return EXIT_SUCCESS;
}

int pex_export(char *fullfilename, char *dstype, char *labels, int numrows, int numcols,
               decimal *dataset, char **ids, struct class_struct *classes)
{
    if (classes->pex_consist == DEFAULT_VALUE) classes->pex_consist = 0;

    // open file in text write mode
    FILE *outfp=fopen(fullfilename,"w");
    if(outfp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    char *thisline = (char*) malloc ((numcols+1) * m_maxfld_sz * sizeof(char));
    char strval[m_maxfld_sz];
    int auxval = 0;
    int nrow = 0;
    int ncol = 0;
    decimal pseudocl = 0.0;

    // header
    fprintf(outfp, "%s\n%d\n%d\n%s\n", dstype, numrows, numcols, labels);

    // class consistency
    if ((classes->values) && (classes->enumeration) && (classes->pex_consist))
        pex_classconsist(classes);

    // data
    for (nrow=0; nrow < numrows; nrow++)
    {
        strcpy(thisline, "");
        if (ids) {
            sprintf(strval, "%s;", *(ids+nrow));
            strcat(thisline, strval);
        } else {
            sprintf(strval, "%d;", nrow);
            strcat(thisline, strval);
        }
        for (ncol=0; ncol < numcols; ncol++)
        {
            auxval = nrow * numcols + ncol;
            sprintf(strval, "%g", *(dataset + auxval));
            strcat(thisline, strval);
            if (ncol < numcols-1) strcat(thisline, ";");
        }
        if (classes->values) {
            sprintf(strval, ";%s", *(classes->values + nrow));
            strcat(thisline, strval);
        } else {
            pseudocl = (1.0/numrows) * nrow;
            sprintf(strval, ";%g", pseudocl);
            strcat(thisline, strval);
        }
        fprintf(outfp, "%s\n", thisline);
    }

    fclose(outfp);
    free(thisline);

    return EXIT_SUCCESS;
}

int pex_classconsist(struct class_struct *classes)
{
    if ((!classes->values) || (!classes->enumeration))
        return ERR_INVALID_PARAMETERS;

    unsigned char isconsist = 1;
	char clvalue[m_maxfld_sz];   // class value
	size_t cllen;                // length of class value
	int index = 0;
    int i, j;

    // consistency check
    for (i=0; i < classes->enum_size; i++)
    {
        cllen = strlen(*(classes->enumeration+i));
        for (j=0; j < cllen; j++)
        {
            if (isalpha((*(classes->enumeration+i))[j])) {
                isconsist = 0;
                break;
            }
        }
        if (!isconsist) break;
    }
    if (isconsist) return EXIT_SUCCESS;

    // change values
    for (i=0; i < classes->size; i++)
    {
        index = charptr_getposition(classes->enumeration, classes->enum_size, *(classes->values+i));
        sprintf(clvalue, "%d", index);
        strcpy(*(classes->values+i), clvalue);
    }

    return EXIT_SUCCESS;
}

int weka_importheader(char *fullfilename, char *relation, char **attributes, int *numrows,
                      int *numcols, int *startrow, unsigned char *hasid, unsigned char *hascl)
{
    FILE *infp=fopen(fullfilename,"r");
    if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

	int thisline_sz = m_maxflds * m_maxfld_sz * sizeof(char);
	char *thisline = (char*) malloc(thisline_sz);
	char *thisline_lwr;
	char *thisline_;
	char **datarow = (char**) malloc (1*sizeof(char*));

	int startrow_=0;
	*startrow=0;
	*numrows=0;
    *numcols=0;
    if (hasid) *hasid=0;
    if (hascl) *hascl=0;

	while(fgets(thisline,thisline_sz,infp)!=0)
	{
	    *numrows+=1;

        if (startrow_==0)
        {
            thisline_ = strdup(thisline);
            thisline_lwr = strdup(thisline);
            str2lwr(thisline_lwr, thisline);

            if (relation && (!(strncmp(thisline_lwr, "@relation", 9))))
            {
                memset(thisline, 32, 9);
                thisline_ = strrep(thisline, "'", "");
                strrspc(relation, thisline_);
            }
            if (!(strncmp(thisline_lwr, "@attribute", 10)))
            {
                thisline_ = strrep(thisline, "\t", ",");
                thisline_ = strrep(thisline_, " ", ",");
                thisline_ = strrep(thisline_, "'", "");

                parse(thisline_, ",", NULL, datarow, 1, 1);
                strcpy(thisline, *datarow);

                str2lwr(thisline_,thisline);
                if (!(strncmp(thisline_, "id", 2)))
                {
                    if (hasid) *hasid = 1;
                }
                else if (!(strncmp(thisline_, "class", 5)))
                {
                    if (hascl) *hascl = 1;
                }
                else
                {
                    if (attributes) {
                        *(attributes + *numcols) = (char*) malloc(m_maxfld_sz * sizeof(char));
                        strcpy(*(attributes + *numcols), thisline);
                    }
                    *numcols+=1;
                }
            }
            if (!(strncmp(thisline_lwr, "@data", 5)))
            {
                startrow_ = *numrows+1;
            }

            free(thisline_);
            free(thisline_lwr);
        }
        else if ((startrow_>0) && (*startrow==0))
        {
            thisline_ = strdup(thisline);
            strrspc(thisline, thisline_);

            if (!(thisline[0]))
                startrow_+=1;
            else
                *startrow=startrow_;

            free(thisline_);
        }
	}

    // add null char to the attributes pointer
    if (attributes) {
        *(attributes + *numcols) = (char*) malloc(1 * sizeof(char));
        *(attributes + *numcols) = NULL;
    }

    *numrows-= *(startrow)-1;
    fclose(infp);

    // release memory
    free(datarow);
    free(thisline);

    return EXIT_SUCCESS;
}

int weka_importdata(char *fullfilename, int numrows, int numcols, int startrow,
                    decimal *dataset, struct id_struct *ids, struct class_struct *classes)
{
    if (ids->retrieve == DEFAULT_VALUE) ids->retrieve = NO_ID_IN_FILE;
    if (ids->size == DEFAULT_VALUE) ids->size = numrows;
    if (classes->retrieve == DEFAULT_VALUE) classes->retrieve = NO_CLASS_IN_FILE;
    if (classes->size == DEFAULT_VALUE) classes->size = numrows;

    // exist id and class in file?
	int idcol, clcol;
	idcol = ((ids->retrieve == NO_ID_IN_FILE) || (ids->retrieve == CREATE_ID)) ? 0 : 1;
	clcol = (classes->retrieve == NO_CLASS_IN_FILE) ? 0 : 1;

    int thisline_sz = m_maxfld_sz * (numcols+idcol+clcol) * sizeof(char);
	char *thisline = (char*) malloc(thisline_sz);
	char **datarow = (char**) malloc((numcols+idcol+clcol) * sizeof(char*));
	int fldcnt = 0;
	int recordcnt = 0;
	char *clvalue;     // current class value
	size_t cllen;      // length of class value

	FILE *infp=fopen(fullfilename,"r");
	if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    // jump the header
    int i=0;
	while(fgets(thisline,thisline_sz,infp)!=0) {
	    i++;
        if (i >= startrow-1) break;
	}

    // get dataset
	while(fgets(thisline,thisline_sz,infp)!=0)
	{
        parse(thisline,",",&fldcnt,datarow,0,0);

        if (ids->retrieve == GET_ID) {
            *(ids->values + recordcnt) = (char*) malloc((strlen(*datarow)+1) * sizeof(char));
            strcpy(*(ids->values + recordcnt), *datarow);
        }
        else if (ids->retrieve == CREATE_ID) {
            *(ids->values + recordcnt) = (char*) malloc((strlen(*datarow)+1) * sizeof(char));
            sprintf(*(ids->values + recordcnt), "%d", recordcnt+1);
        }

        for(i=0;i<fldcnt-idcol-clcol;i++)
            *(dataset+i+recordcnt*numcols) = atof(*(datarow+i+idcol));

        if (classes->retrieve == GET_CLASS) {
            cllen = strlen(*(datarow+fldcnt-1));
            clvalue = (char*) malloc(cllen * sizeof(char));
            strrspc(clvalue, *(datarow+fldcnt-1));

            *(classes->values + recordcnt) = (char*) malloc(cllen * sizeof(char));
            strcpy(*(classes->values + recordcnt), clvalue);

            if ((classes->enum_size < m_maxclass) && (classes->enum_size < numrows-1))
                classes->enum_size = charptr_distinctvaladd(classes->enumeration, clvalue);
            else
                return ERR_WEKA_FILE_INVALID;

            free(clvalue);
        }

        recordcnt++;
	}

    fclose(infp);

    // release memory
    free(datarow);
    free(thisline);

    return EXIT_SUCCESS;
}

int weka_export(char *fullfilename, char *relation, char **attributes, int numrows, int numcols,
                decimal *dataset, char **ids, struct class_struct *classes)
{
    // open file in text write mode
    FILE *outfp=fopen(fullfilename,"w");
    if(outfp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    char *thisline = (char*) malloc ((numcols+1) * m_maxfld_sz * sizeof(char));
    char strval[m_maxfld_sz];
    int auxval=0;
    int nrow=0;
    int ncol=0;
    int nclass=0;

    // header
    fprintf(outfp, "\%% Generated by %s in %s", PRGNAME, datetime_get());
    fprintf(outfp, "\%% Author: %s\n", "Paulo Joia Filho");
    fprintf(outfp, "\%% instances: %d\n", numrows);
    fprintf(outfp, "\%% dimension: %d\n", numcols);
    fprintf(outfp, "\%% classes:   %d\n\n", classes->enum_size);

    fprintf(outfp, "@RELATION %s\n\n", relation);

    // attribute: id and fields
    if (ids)
    {
        fprintf(outfp, "@ATTRIBUTE\t%s\tSTRING\n", "id");
    }
    for (ncol=0; ncol < numcols; ncol++)
    {
        fprintf(outfp, "@ATTRIBUTE\t%s\tREAL\n", *(attributes+ncol));
    }

    // attribute: class
    strcpy(thisline, "");
    if ((classes->values) && (classes->enumeration))
    {
        strcpy(thisline, "@ATTRIBUTE\tclass {");
        for (nclass=0; nclass < classes->enum_size; nclass++)
        {
            sprintf(strval, "%s", *(classes->enumeration + nclass));
            strcat(thisline, strval);

            if (nclass < classes->enum_size-1)
                strcat(thisline, ",");
            else
                strcat(thisline, "}\n");
        }
    }
    fprintf(outfp, "%s\n", thisline);
    strcpy(thisline, "");

    // data
    fprintf(outfp, "@DATA\n");
    for (nrow=0; nrow < numrows; nrow++)
    {
        if (ids)
        {
            sprintf(strval, "%s,", *(ids+nrow));
            strcat(thisline, strval);
        }
        for (ncol=0; ncol < numcols; ncol++)
        {
            auxval = nrow * numcols + ncol;
            sprintf(strval, "%g", *(dataset + auxval));
            strcat(thisline, strval);
            if (ncol < numcols-1) strcat(thisline, ",");
        }
        if ((classes->values) && (classes->enumeration))
        {
            sprintf(strval, ",%s", *(classes->values + nrow));
            strcat(thisline, strval);
        }
        fprintf(outfp, "%s\n", thisline);
        strcpy(thisline, "");
    }

    fclose(outfp);
    free(thisline);

    return EXIT_SUCCESS;
}

void parse(char *record, char *delim, int *fldcnt, char **datarow, int startcol, int endcol)
{
    /*
        Take a care!!!
        The variable "record" will be destroyed by "strtok" function.
    */
    char *p = NULL;
    int fld  = 0;
    int fld_ = 0;

    p = strtok(record, delim);
    while(p != NULL)
    {
        if (datarow)
        {
            if ( ((fld >= startcol) && (fld <= endcol)) || ((startcol==0) && (endcol==0)) )
            {
                *(datarow+fld_) = (char*) malloc((strlen(p)+1)*sizeof(char));
                strcpy(*(datarow+fld_), p);
                fld_++;
            }
        }
        fld++;
        p = strtok(NULL, delim);
	}

	if (fldcnt)
        *fldcnt = fld;
}
