#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "structmember.h"

#define MAX_ORDER			15

#define PYIIR1_PASS_LP		1
#define PYIIR1_PASS_HP		2
#define PYIIR1_PASS_BP		3

#define PYIIR1_TYPE_NULL 		0
#define PYIIR1_TYPE_BUTTERWORTH 	1

#define xstr(s) #s
#define str(s) xstr(s)


#ifndef Py_TYPE
    #define Py_TYPE(ob) (((PyObject*)(ob))->ob_type)
#endif

#define eprintf // printf

extern char errbuffer[4096];
extern PyObject *pyiir1Error;

#define pyiir1_HEAD \
	double lpass;		/* Low Pass freq */ \
    double hpass;		/* High Pass Freq */ \
    float srate; \
    int type;			/* Type of the filter */ \
    int btype;			/* Band Type of the filter */ \
    int order; \
    char description[256]; \
    void * iir1;                            /* iir object */ \
    double (* filter)(void *, double);      /* filter function */ \
    void (*reset)(void *);                  /* reset funciton */ \
    void (*delfun)(void *);                  /* delete funciton */ \


#define FUNC_DEALLOC(X) \
static void X##Filter_dealloc(pyiir1_Filter##X * self) { \
	eprintf("DEBUG:: Dealloc\n"); \
	if (self->iir1 != NULL) { \
		self->delfun(self->iir1); \
	} \
    Py_TYPE(self)->tp_free((PyObject*)self); \
} \

#define FUNC_NEW(X) \
static PyObject * X##Filter_new(PyTypeObject *type, PyObject *args, PyObject *kwds) { \
	eprintf("DEBUG:: New\n"); \
    pyiir1_Filter##X *self; \
    self = (pyiir1_Filter##X *)type->tp_alloc(type, 0); \
    if (self != NULL) { \
    	self->lpass = 0.0; \
    	self->hpass = 0.0; \
    	self->srate = 0.0; \
    	self->type = PYIIR1_TYPE_NULL; \
    	self->btype = PYIIR1_TYPE_NULL; \
    	self->order = 0; \
    	self->iir1 = NULL; \
    	sprintf(self->description, #X); \
    } \
    return (PyObject *)self; \
} \

#define INNER_FUNC_RESET(N, C, O) \
    static void inner_reset_##N##_##O(void * iir) { \
        (( C < O > * )iir)->reset(); \
    } \

#define INNER_FUNC_FILTER(N, C, O) \
    static double inner_filter_##N##_##O(void * iir, double sample) { \
        return ((C < O > *)iir)->filter(sample); \
    } \

#define INNER_FUNC_DELETE(N, C, O) \
    static void inner_delete_##N##_##O(void * iir) { \
        delete ((C < O > *)iir); \
    } \


#define DEFORD1(N, C) INNER_FUNC_RESET(N, C, 1);  INNER_FUNC_FILTER(N, C, 1);  INNER_FUNC_DELETE(N, C, 1);
#define DEFORD2(N, C) INNER_FUNC_RESET(N, C, 2);  INNER_FUNC_FILTER(N, C, 2);  INNER_FUNC_DELETE(N, C, 2);  DEFORD1(N, C)
#define DEFORD3(N, C) INNER_FUNC_RESET(N, C, 3);  INNER_FUNC_FILTER(N, C, 3);  INNER_FUNC_DELETE(N, C, 3);  DEFORD2(N, C)
#define DEFORD4(N, C) INNER_FUNC_RESET(N, C, 4);  INNER_FUNC_FILTER(N, C, 4);  INNER_FUNC_DELETE(N, C, 4);  DEFORD3(N, C)
#define DEFORD5(N, C) INNER_FUNC_RESET(N, C, 5);  INNER_FUNC_FILTER(N, C, 5);  INNER_FUNC_DELETE(N, C, 5);  DEFORD4(N, C)
#define DEFORD6(N, C) INNER_FUNC_RESET(N, C, 6);  INNER_FUNC_FILTER(N, C, 6);  INNER_FUNC_DELETE(N, C, 6);  DEFORD5(N, C)
#define DEFORD7(N, C) INNER_FUNC_RESET(N, C, 7);  INNER_FUNC_FILTER(N, C, 7);  INNER_FUNC_DELETE(N, C, 7);  DEFORD6(N, C)
#define DEFORD8(N, C) INNER_FUNC_RESET(N, C, 8);  INNER_FUNC_FILTER(N, C, 8);  INNER_FUNC_DELETE(N, C, 8);  DEFORD7(N, C)
#define DEFORD9(N, C) INNER_FUNC_RESET(N, C, 9);  INNER_FUNC_FILTER(N, C, 9);  INNER_FUNC_DELETE(N, C, 9);  DEFORD8(N, C)
#define DEFORD10(N, C) INNER_FUNC_RESET(N, C, 10); INNER_FUNC_FILTER(N, C, 10); INNER_FUNC_DELETE(N, C, 10); DEFORD9(N, C)
#define DEFORD11(N, C) INNER_FUNC_RESET(N, C, 11); INNER_FUNC_FILTER(N, C, 11); INNER_FUNC_DELETE(N, C, 11); DEFORD10(N, C)
#define DEFORD12(N, C) INNER_FUNC_RESET(N, C, 12); INNER_FUNC_FILTER(N, C, 12); INNER_FUNC_DELETE(N, C, 12); DEFORD11(N, C)
#define DEFORD13(N, C) INNER_FUNC_RESET(N, C, 13); INNER_FUNC_FILTER(N, C, 13); INNER_FUNC_DELETE(N, C, 13); DEFORD12(N, C)
#define DEFORD14(N, C) INNER_FUNC_RESET(N, C, 14); INNER_FUNC_FILTER(N, C, 14); INNER_FUNC_DELETE(N, C, 14); DEFORD13(N, C)
#define DEFORD15(N, C) INNER_FUNC_RESET(N, C, 15); INNER_FUNC_FILTER(N, C, 15); INNER_FUNC_DELETE(N, C, 15); DEFORD14(N, C)

#define BINORD(N, C, O, ...) \
    if (order == O) { \
        C<O> * filter = new C<O>; \
        filter->setup(order, srate, __VA_ARGS__); \
        self->iir1 = (void *) filter; \
        self->reset = &inner_reset_##N##_##O; \
        self->delfun = &inner_delete_##N##_##O; \
        self->filter = &inner_filter_##N##_##O; \
    } \

#define BINORD0(N, C, ...) BINORD(N, C, 0, __VA_ARGS__)
#define BINORD1(N, C, ...) BINORD(N, C, 1, __VA_ARGS__)
#define BINORD2(N, C, ...) BINORD(N, C, 2, __VA_ARGS__) BINORD1(N, C, __VA_ARGS__)
#define BINORD3(N, C, ...) BINORD(N, C, 3, __VA_ARGS__) BINORD2(N, C, __VA_ARGS__)
#define BINORD4(N, C, ...) BINORD(N, C, 4, __VA_ARGS__) BINORD3(N, C, __VA_ARGS__)
#define BINORD5(N, C, ...) BINORD(N, C, 5, __VA_ARGS__) BINORD4(N, C, __VA_ARGS__)
#define BINORD6(N, C, ...) BINORD(N, C, 6, __VA_ARGS__) BINORD5(N, C, __VA_ARGS__)
#define BINORD7(N, C, ...) BINORD(N, C, 7, __VA_ARGS__) BINORD6(N, C, __VA_ARGS__)
#define BINORD8(N, C, ...) BINORD(N, C, 8, __VA_ARGS__) BINORD7(N, C, __VA_ARGS__)
#define BINORD9(N, C, ...) BINORD(N, C, 9, __VA_ARGS__) BINORD8(N, C, __VA_ARGS__)
#define BINORD10(N, C, ...) BINORD(N, C, 10, __VA_ARGS__) BINORD9(N, C, __VA_ARGS__)
#define BINORD11(N, C, ...) BINORD(N, C, 11, __VA_ARGS__) BINORD10(N, C, __VA_ARGS__)
#define BINORD12(N, C, ...) BINORD(N, C, 12, __VA_ARGS__) BINORD11(N, C, __VA_ARGS__)
#define BINORD13(N, C, ...) BINORD(N, C, 13, __VA_ARGS__) BINORD12(N, C, __VA_ARGS__)
#define BINORD14(N, C, ...) BINORD(N, C, 14, __VA_ARGS__) BINORD13(N, C, __VA_ARGS__)
#define BINORD15(N, C, ...) BINORD(N, C, 15, __VA_ARGS__) BINORD14(N, C, __VA_ARGS__)

#define FUNC_INIT_DEF(X) \
static int X##Filter_init(pyiir1_Filter##X *self, PyObject *args, PyObject *kwds) { \
	eprintf("DEBUG:: Init\n"); \
	double hpass = 0.0; \
	double lpass = 0.0; \
	double srate = 0.0; \
	int type = 0; \
	int btype = 0; \
	int order = 0; \

#define FUNC_INIT_DEBUG \
	eprintf("DEBUG:: order %d, srate %f\n", order, srate); \


#define FUNC_INITKWARGS_LIST(...) \
    static char *kwlist[] = {"order", "srate", __VA_ARGS__, NULL}; \

#define FUNC_INITWKARGS_PARSE(format, ...) \
    if (! PyArg_ParseTupleAndKeywords(args, kwds, "id"format, kwlist,  \
                                      &order, &srate, __VA_ARGS__)) { \
        return -1;  \
    } \
    if (order > MAX_ORDER) { \
    	sprintf(errbuffer, "Order (%d) cannot be more than %d", order, MAX_ORDER); \
    	PyErr_SetString(pyiir1Error, errbuffer); \
    	return -1; \
    }  \

#define FUNC_INIT_ERROR(cond, msg, ...) \
    if (cond) { \
    	sprintf(errbuffer, msg,##__VA_ARGS__); \
    	PyErr_SetString(pyiir1Error, errbuffer); \
    	return -1; \
    }  \

#define FUNC_INIT_INIT(N, C, ...) \
    eprintf("DEBUG:: Init INIT\n"); \
    BINORD15(N, C, __VA_ARGS__) \

#define FUNC_INIT_FINALIZE \
    if (hpass > 0 && lpass > hpass) { \
    	btype = PYIIR1_PASS_BP; \
    } else if (hpass > 0 && lpass == 0) { \
    	btype = PYIIR1_PASS_HP; \
    } else if (hpass == 0 && lpass > 0) { \
    	btype = PYIIR1_PASS_LP; \
    } else { \
    	sprintf(errbuffer, "This should never happen"); \
    	PyErr_SetString(pyiir1Error, errbuffer); \
    	return -1; \
    } \
    self->hpass = hpass; \
    self->lpass = lpass; \
    self->type = type; \
    self->btype = btype; \
    self->order = order; \
    self->srate = srate; \
    return 0; \
}

#define FUNC_RESET(X) \
static PyObject * X##Filter_reset(pyiir1_Filter##X * self) { \
	eprintf("DEBUG:: Reset\n"); \
	self->reset(self->iir1); \
	Py_RETURN_NONE; \
} \

#define FUNC_FILTER(X, ...) \
static PyObject * X##Filter_filter(pyiir1_Filter##X * self, PyObject *args) { \
	eprintf("DEBUG:: Filter\n"); \
    PyListObject *elems = NULL; \
    if (!PyArg_ParseTuple(args, "O", &elems)) { \
    	sprintf(errbuffer, "Cannot parse list"); \
    	PyErr_SetString(pyiir1Error, errbuffer); \
        return NULL; \
    } \
    int n_elems = PyList_GET_SIZE(elems); \
    double elem, elem2; \
    for (int i = 0; i < n_elems; i++) { \
    	elem = PyFloat_AS_DOUBLE(PyList_GET_ITEM(elems, i)); \
        elem2 = self->filter(self->iir1, elem); \
    	PyList_SET_ITEM(elems, i, PyFloat_FromDouble(elem2)); \
    } \
    Py_RETURN_NONE; \
} \

#define FUNC_STR(X) \
static PyObject * X##Filter_str(pyiir1_Filter##X * self) { \
	static PyObject *format = NULL; \
    PyObject *args, *result; \
    args = NULL; \
    result = NULL; \
    if (format == NULL) { \
        format = PyUnicode_FromString("%s (order %d) [%f %f]"); \
        if (format == NULL) { \
            return NULL; \
        } \
    } \
    args = Py_BuildValue("siff", self->description, self->order, self->hpass, self->lpass); \
    result = PyUnicode_Format(format, args); \
    Py_DECREF(args); \
    return result; \
} \

#define MEMBER_DEF(X) \
static PyMemberDef X##Filter_members[] = { \
    {"lpass", T_DOUBLE, offsetof(pyiir1_Filter##X, lpass), 0, "low pass band"}, \
    {"hpass", T_DOUBLE, offsetof(pyiir1_Filter##X, hpass), 0, "high pass band"}, \
    {"srate", T_DOUBLE, offsetof(pyiir1_Filter##X, srate), 0, "high pass band"}, \
    {"type", T_INT, offsetof(pyiir1_Filter##X, type), 0, "type of filter"}, \
    {"btype", T_INT, offsetof(pyiir1_Filter##X, btype), 0, "band type of filter"}, \
    {"order", T_INT, offsetof(pyiir1_Filter##X, order), 0, "order of filter"}, \
    {"description", T_STRING, offsetof(pyiir1_Filter##X, description), 0, "description of filter"}, \
    {NULL}  /* Sentinel */ \
}; \

#define METHOD_DEF(X) \
static PyMethodDef X##Filter_methods[] = { \
	{"reset", (PyCFunction)X##Filter_reset, METH_NOARGS, "Reset the state of the filter"}, \
	{"filter", (PyCFunction)X##Filter_filter, METH_VARARGS, "Filter a list of samples"}, \
    {NULL}  /* Sentinel */ \
}; \

#define OBJECT_DEF(X) \
PyTypeObject pyiir1_filter ##X## Type = { \
    PyVarObject_HEAD_INIT(NULL, 0) \
    /*0,*/                         	 /*ob_size*/ \
    "pyiir1.Filter"#X,             /*tp_name*/ \
    sizeof(pyiir1_Filter##X), /*tp_basicsize*/ \
    0,                         	 /*tp_itemsize*/ \
    (destructor) X##Filter_dealloc,                         /*tp_dealloc*/ \
    0,                         /*tp_print*/ \
    0,                         /*tp_getattr*/ \
    0,                         /*tp_setattr*/ \
    0,                         /*tp_compare*/ \
    (reprfunc) X##Filter_str,                /*tp_repr*/ \
    0,                         /*tp_as_number*/ \
    0,                         /*tp_as_sequence*/ \
    0,                         /*tp_as_mapping*/ \
    0,                         /*tp_hash */ \
    0,                         /*tp_call*/ \
    (reprfunc) X##Filter_str,                /*tp_str*/ \
    0,                         /*tp_getattro*/ \
    0,                         /*tp_setattro*/ \
    0,                         /*tp_as_buffer*/ \
    Py_TPFLAGS_DEFAULT,        /*tp_flags*/ \
    "Pyiir1 Filter " #X "objects",           /* tp_doc */ \
    0,		               /* tp_traverse */ \
    0,		               /* tp_clear */ \
    0,		               /* tp_richcompare */ \
    0,		               /* tp_weaklistoffset */ \
    0,		               /* tp_iter */ \
    0,		               /* tp_iternext */ \
    X##Filter_methods,             /* tp_methods */ \
    X##Filter_members,            /* tp_members */ \
    0,                         /* tp_getset */ \
    0,                         /* tp_base */ \
    0,                         /* tp_dict */ \
    0,                         /* tp_descr_get */ \
    0,                         /* tp_descr_set */ \
    0,                         /* tp_dictoffset */ \
    (initproc) X##Filter_init,      /* tp_init */ \
    0,                         /* tp_alloc */ \
    X##Filter_new,                 /* tp_new */ \
}; \

#define INIT_OBJECT(X) \
	pyiir1_filter ##X## Type.tp_new = PyType_GenericNew; \
	if (PyType_Ready(&pyiir1_filter ##X## Type) < 0) { \
        return NULL; \
    } \

#define ADD_OBJECT(X) \
    Py_INCREF(&pyiir1_filter ##X## Type); \
    PyModule_AddObject(m, #X, (PyObject *)&pyiir1_filter ##X## Type); \

#endif /* __CONFIG_H__ */
