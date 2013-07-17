/*
* Copyright 2002 Sun Microsystems, Inc.,
* Copyright 2002 University Of Toronto
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Library General Public
* License as published by the Free Software Foundation; either
* version 2 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Library General Public License for more details.
*
* You should have received a copy of the GNU Library General Public
* License along with this library; if not, write to the
* Free Software Foundation, Inc., 59 Temple Place - Suite 330,
* Boston, MA 02111-1307, USA.
*/
#include "pan_useful.h"

/*
  constants
*/
#define ARRAYLIST_INITIAL_CAPACITY 10
#define ARRAYLIST_CAPACITY_DELTA 10

static const size_t m_object_size = sizeof(Object);

/*
  structures
*/
struct arraylist_struct {
  int _current_capacity;
  Object *_data;
  int _size;
  const boolean (*_equals)();
};

/*
  private function declarations
*/
static void *checked_malloc(const size_t size);


void arraylist_free(const arraylist list)
{
  free(list->_data);
  free(list);
}

arraylist arraylist_create(const boolean (*equals)(const Object object_1, const Object object_2))
{
  arraylist list;

  list = checked_malloc(sizeof(struct arraylist_struct));
  list->_current_capacity = ARRAYLIST_INITIAL_CAPACITY;
  list->_data = checked_malloc(m_object_size * list->_current_capacity);
  list->_size = 0;
  list->_equals = equals;

  return list;
}

boolean arraylist_add(const arraylist list, Object object)
{
  int old_size = arraylist_size(list);
  int new_capacity;
  Object *new_data;

  (list->_size)++;
  if (old_size == list->_current_capacity)
    {
      new_capacity = list->_current_capacity + ARRAYLIST_CAPACITY_DELTA;
      new_data = checked_malloc(m_object_size * new_capacity);
      memcpy(new_data, list->_data, m_object_size * old_size);
      free(list->_data);
      (list->_data) = new_data;
      list->_current_capacity = new_capacity;
    }
  (list->_data)[old_size] = object;
  return True;
}

boolean arraylist_remove(const arraylist list, const Object object)
{
  int length = arraylist_size(list);
  int last_index = length - 1;
  int new_size, new_capacity;
  int index;

  for (index = 0; index < length; index++)
    {
      if ((*list->_equals)(arraylist_get(list, index), object))
	{
	  (list->_size)--;
	  if (index < last_index)
	    {
	      memmove(list->_data + index, list->_data + index + 1, m_object_size * (last_index - index));
	      new_size = list->_size;
	      new_capacity = list->_current_capacity - ARRAYLIST_CAPACITY_DELTA;
	      if (new_capacity > new_size)
		{
		  list->_data = realloc(list->_data, m_object_size * new_capacity);
		  list->_current_capacity = new_capacity;
		}
	    }
	  return True;
	}
    }
  return False;
}

boolean arraylist_contains(const arraylist list, const Object object)
{
  return (arraylist_index_of(list, object) > -1);
}

int arraylist_index_of(const arraylist list, const Object object)
{
  int length = arraylist_size(list);
  int index;

  for (index = 0; index < length; index++)
    {
      if ((*list->_equals)(arraylist_get(list, index), object))
	{
	  return index;
	}
    }
  return -1;
}

boolean arraylist_is_empty(const arraylist list)
{
  return (0 == arraylist_size(list));
}

int arraylist_size(const arraylist list)
{
  return list->_size;
}

Object arraylist_get(const arraylist list, const int index)
{
  return list->_data[index];
}

void arraylist_clear(const arraylist list)
{
  list->_data = realloc(list->_data, m_object_size * ARRAYLIST_INITIAL_CAPACITY);
  list->_current_capacity = ARRAYLIST_INITIAL_CAPACITY;
  list->_size = 0;
}

void arraylist_sort(const arraylist list, const int (*compare)(const Object object_1, const Object object_2))
{
  qsort(list->_data,
	arraylist_size(list),
	sizeof(Object),
	(int (*)())compare);
}

static void *checked_malloc(const size_t size)
{
  void *data;

  data = malloc(size);
  if (data == NULL)
    {
      fprintf(stderr, "\nOut of memory.\n");
      fflush(stderr);
      exit(EXIT_FAILURE);
    }

  return data;
}
