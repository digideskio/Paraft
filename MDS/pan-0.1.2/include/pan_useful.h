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
#ifndef PAN_USEFUL_H_INCLUDED
#define PAN_USEFUL_H_INCLUDED

/******************************************************************************/
/*                                   INCLUDES                                 */
/******************************************************************************/
#include "pan_err.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include <float.h>
#include <assert.h>
#include <string.h>
#include <ctype.h>

/******************************************************************************/
/*                                   CONSTANTS                                */
/******************************************************************************/
#define DBL_PRECISION 0 // use always 0=float, 1=double

#define NULL_INT 0
#define NULL_DEC 0.0
#define DEFAULT_VALUE 0
#define EPSILON 1.0E-7
#define True 1
#define False 0

#define DEBUG_INFO(title) ({ \
    char strinfo[1024]; \
    sprintf(strinfo,"Title: %s \nDate/Time: %s - %s \nFile: %s \nLine: %d \n", \
            title,__DATE__,__TIME__,__FILE__,__LINE__); \
    strinfo; \
})

/******************************************************************************/
/*                                    TYPES                                   */
/******************************************************************************/
#if DBL_PRECISION
    typedef double decimal;
    #define DECIMAL_PRINT "%15.16f"
    #define DEC_MAX DBL_MAX
    #define DEC_MIN DBL_MIN
#else
    typedef float decimal;
    #define DECIMAL_PRINT "%10.8f"
    #define DEC_MAX FLT_MAX
    #define DEC_MIN FLT_MIN
#endif

#undef boolean
#define boolean short unsigned int

#undef Object
#define Object void*

/******************************************************************************/
/*                         ENUMERATIONS / STRUCTURES                          */
/******************************************************************************/
typedef struct arraylist_struct *arraylist;


/******************************************************************************/
/*                                   FUNCTIONS                                */
/******************************************************************************/
// Get precision type
unsigned char precision_get();

// Get date and time
char* datetime_get();

// Replace value in the string
char* strrep (char const *original,char const *pattern, char const *replacement);

// Remove space \t \n of string
char* strrspc(char *trg, char *src);

// Convert string to lower case
char* str2lwr(char *trg, char *src);

// Join pointer values in a single string separated by delim
// Notice that last element in src must be a null char
int strjoin(char *trg, char **src, char *delim);

// Add only distinct values to one pointer previously allocated with calloc
int charptr_distinctvaladd(char **ptr, char *value);

// Get the value index in a char pointer
int charptr_getposition(char **ptr, int sz, char *value);

// Alloc values to decimal pointer
inline void decptrcpy(decimal *trg, int tinipos, int tfinpos, decimal *src, int sinipos, int sfinpos);

// Clear the decimal pointer content without deallocate memory
void decptrclr(decimal *ptr, int sz);

// Read text file and print its content
void textfile_print(char *fullfilename);

// Print a horizontal line of len length
void hzline_print(int len);

// Print one string on the centerline
void centerline_print(char *str, int linelen);

/*
  Arraylist handling
*/
void arraylist_free(const arraylist list);
arraylist arraylist_create(const boolean (*equals)(const Object object_1, const Object object_2));
boolean arraylist_add(const arraylist list, Object object);
boolean arraylist_remove(const arraylist list, const Object object);
boolean arraylist_contains(const arraylist list, const Object object);
int arraylist_index_of(const arraylist list, const Object object);
boolean arraylist_is_empty(const arraylist list);
int arraylist_size(const arraylist list);
Object arraylist_get(const arraylist list, const int index);
void arraylist_clear(const arraylist list);
void arraylist_sort(const arraylist list, const int (*compare)(const Object object_1, const Object object_2));

#endif // PAN_USEFUL_H_INCLUDED
