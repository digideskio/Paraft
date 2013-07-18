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
#ifndef PAN_DCONV_H_INCLUDED
#define PAN_DCONV_H_INCLUDED

/******************************************************************************/
/*                                   INCLUDES                                 */
/******************************************************************************/
#include "pan_useful.h"


/******************************************************************************/
/*                                   DEFINES                                  */
/******************************************************************************/
#define FIELD_NAME(i) ({         \
    char name[15];               \
    sprintf(name, "attr%d", i);  \
    name;                        \
})


/******************************************************************************/
/*                         ENUMERATIONS / STRUCTURES                          */
/******************************************************************************/
enum id_retrieve_enum {
    GET_ID = 1,
    DO_NOT_GET_ID,  // id is in file
    NO_ID_IN_FILE,
    CREATE_ID       // id is not in file
};
enum class_retrieve_enum {
    GET_CLASS = 1,
    DO_NOT_GET_CLASS,
    NO_CLASS_IN_FILE
};

struct id_struct {
    char **values;                     // allocated
    int size;                          // optional: DEFAULT_VALUE = NULL_INT
    enum id_retrieve_enum retrieve;    // optional
};
struct class_struct {
    char **values;                     // allocated
    int size;                          // optional: DEFAULT_VALUE = NULL_INT
    enum class_retrieve_enum retrieve; // optional
    char **enumeration;                // allocated
    int enum_size;                     // optional: DEFAULT_VALUE = NULL_INT
    unsigned char pex_consist;         // optional: DEFAULT_VALUE = 0
};

/******************************************************************************/
/*                                   FUNCTIONS                                */
/******************************************************************************/
// Field size handling
inline unsigned short int maxflds_get();
inline void maxflds_set(unsigned short int value);
inline unsigned char maxfld_sz_get();
inline void maxfld_sz_set(unsigned char value);
unsigned short int maxclass_get();
void maxclass_set(unsigned short int value);

// Comma separated values handling
int csv_getdimension(char *fullfilename, char *delim, int *numrows, int *numcols,
                     int *startrow, int *endrow, int *startcol, int *endcol);
int csv_importdata(char *fullfilename, char *delim, int numrows, int numcols,
                   int startrow, int endrow, int startcol, int endcol, char **dataset);

// Write pointers to the disk
int charptr_write(char *fullfilename, char **dataset, char *delim, int numrows,
                  int numcols, char insertid, char* textbefore);
int decptr_write(char *fullfilename, decimal *dataset, char *delim, int numrows,
                 int numcols, char insertid, char* textbefore);

// PEx file format handling
int pex_importheader(char *fullfilename, char *dstype, char *labels, int *numrows,
                     int *numcols, unsigned char *isconsist);
int pex_importdata(char *fullfilename, int numrows, int numcols, decimal *dataset,
                   struct id_struct *ids, struct class_struct *classes);
int pex_export(char *fullfilename, char *dstype, char *labels, int numrows, int numcols,
               decimal *dataset, char **ids, struct class_struct *classes);
int pex_classconsist(struct class_struct *classes);
char* pex_labelsconsist(char *labels, int numcols);
char* pex_labelscreate(char *labels, int numcols);

// Weka file format handling
int weka_importheader(char *fullfilename, char *relation, char **attributes, int *numrows,
                      int *numcols, int *startrow, unsigned char *hasid, unsigned char *hascl);
int weka_importdata(char *fullfilename, int numrows, int numcols, int startrow,
                    decimal *dataset, struct id_struct *ids, struct class_struct *classes);
int weka_export(char *fullfilename, char *relation, char **attributes, int numrows, int numcols,
                decimal *dataset, char **ids, struct class_struct *classes);

// Functions that allow command line calling
int pex2weka(char *fullfilename, char *newfilename);
int weka2pex(char *fullfilename, char *newfilename);

// Other
void parse(char *record, char *delim, int *fldcnt, char **datarow, int startcol, int endcol);

#endif // PAN_DCONV_H_INCLUDED
