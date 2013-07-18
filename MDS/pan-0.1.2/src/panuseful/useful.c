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
#include "pan_useful.h"

unsigned char precision_get()
{
    // Note: required by Python
    return (unsigned char)DBL_PRECISION;
}

char* datetime_get()
{
    struct tm *ptr;
    time_t lt;

    lt = time(NULL);
    ptr = localtime(&lt);

    char *datetime = (char*) malloc (30*sizeof(char));
    strcpy(datetime, asctime(ptr));

    return datetime;
}

char* strrep (char const *original,char const *pattern,char const *replacement)
{
    size_t const replen = strlen(replacement);
    size_t const patlen = strlen(pattern);
    size_t const orilen = strlen(original);

    size_t patcnt = 0;
    const char * oriptr;
    const char * patloc;

    /* find how many times the pattern occurs in the original string */
    for (oriptr=original; (patloc=strstr(oriptr, pattern)); oriptr=patloc+patlen)
    {
        patcnt++;
    }

    /* allocate memory for the new string */
    size_t const retlen = orilen + patcnt * (replen - patlen);
    char * const returned = (char *) malloc( sizeof(char) * (retlen + 1) );

    if (returned != NULL)
    {
        /* copy the original string, replacing all the occurrences of the pattern */
        char * retptr = returned;
        for (oriptr=original; (patloc=strstr(oriptr,pattern)); oriptr=patloc+patlen)
        {
            size_t const skplen = patloc - oriptr;

            /* copy the section until the occurence of the pattern */
            strncpy(retptr, oriptr, skplen);
            retptr += skplen;

            /* copy the replacement */
            strncpy(retptr, replacement, replen);
            retptr += replen;
        }
        /* copy the rest of the string */
        strcpy(retptr, oriptr);
    }
    return returned;
}

char *strrspc(char *trg, char *src)
{
    strcpy(trg, "");
    char strtmp[2];
    strtmp[1]='\0';

    char c = *src++;
    while(c)
    {
        if (!(isspace(c)))
        {
            strtmp[0] = c;
            strcat(trg, strtmp);
        }
        c = *src++;
    }
    return trg;
}

char *str2lwr(char *trg, char *src)
{
    strcpy(trg, "");
    char strtmp[2];
    strtmp[1]='\0';

    char c = *src++;
    while(c)
    {
        strtmp[0] = tolower(c);
        strcat(trg, strtmp);
        c = *src++;
    }
    return trg;
}

int strjoin(char *trg, char **src, char *delim)
{
    strcpy(trg, "");

    int n=0;
    while(*(src+n)) n++;

    int i=0;
    for(i=0; i<n; i++)
    {
        strcat(trg, *(src+i));
        if(i<n-1) strcat(trg, delim);
    }

    return 0;
}

int charptr_distinctvaladd(char **ptr, char *value)
{
    int pos=0;
    int allow=1;

    while(*(ptr+pos))
    {
        if(!(strcmp(*(ptr+pos),value)))
            allow=0;
        pos++;
    }
    if(allow)
    {
        *(ptr+pos) = (char*) malloc((strlen(value)+1)*sizeof(char));
        strcpy(*(ptr+pos), value);
        pos++;
    }

    return pos;
}

int charptr_getposition(char **ptr, int sz, char *value)
{
    int pos=0;

    for (pos=0; pos < sz; pos++)
    {
        if (strcmp(*(ptr+pos), value) == 0)
            return pos;
    }

    return -1;
}

inline void decptrcpy(decimal *trg, int tinipos, int tfinpos, decimal *src, int sinipos, int sfinpos)
{
    if (!tinipos) tinipos=0;
    if (!tfinpos) tfinpos=sfinpos-sinipos;
    if (!sinipos) sinipos=0;
    if (!sfinpos) sfinpos=tfinpos-tinipos;

    int t=0, s;
    for(s=sinipos; s<=sfinpos; s++)
    {
        *(trg+tinipos+t) = *(src+s);
        t++;
    }
}

void decptrclr(decimal *ptr, int sz)
{
    int i;
    for(i=0; i<sz; i++)
        *(ptr+i) = 0;
}

void textfile_print(char *fullfilename)
{
    FILE *infp=fopen(fullfilename,"rb");
    if(infp==NULL)
	{
		perror("File open error");
		exit(EXIT_FAILURE);
	}

    char ch = getc(infp);
    while(ch != EOF)
	{
	    putchar(ch);
	    ch = getc(infp);
	}

	fclose(infp);
}

void hzline_print(int len)
{
  int i;

  for (i=0; i<len; i++)
    printf("-");

  printf("\n");
}

void centerline_print(char *str, int linelen)
{
    int stringlen = strlen(str);
    int inipos = (linelen-stringlen)/2;
    int finpos = inipos+stringlen;

    int i;
    for (i=0; i<inipos; i++) printf(" ");
    printf("%s", str);
    for (i=finpos; i<linelen; i++) printf(" ");
    printf("\n");
}
