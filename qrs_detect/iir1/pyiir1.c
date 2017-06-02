#ifndef __PYIIR1_MAIN__
#define __PYIIR1_MAIN__

#include <Python.h>
#include <iir.h>
#include "structmember.h"
#include "config.h"
#include "butterworth.h"

char errbuffer[4096];
PyObject *pyiir1Error;

static PyMethodDef Pyiir1Methods[] = {
	{NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION == 3
static struct PyModuleDef pyiir1module = {
   PyModuleDef_HEAD_INIT,
   "pyiir1",   /* name of module */
   NULL, /* module documentation, may be NULL */
   -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
   Pyiir1Methods
};
#endif


#if PY_MAJOR_VERSION == 3
PyMODINIT_FUNC PyInit_pyiir1(void) {
#else
PyMODINIT_FUNC initpyiir1(void) {
#endif
	PyObject *m;
    INIT_OBJECT(ButterworthLowPass)
    INIT_OBJECT(ButterworthHighPass)
	INIT_OBJECT(ButterworthBandPass)
#if PY_MAJOR_VERSION == 3
	m = PyModule_Create(&pyiir1module);
	if (m == NULL) {
		return NULL;
    }
#else
	m = Py_InitModule3("pyiir1", Pyiir1Methods, "IIR1 Python Interface");
	if (m == NULL) {
		return;
    }
#endif

	pyiir1Error = PyErr_NewException("pyiir1.error", NULL, NULL);
	Py_INCREF(pyiir1Error);
	PyModule_AddObject(m, "error", pyiir1Error);

    ADD_OBJECT(ButterworthLowPass)
    ADD_OBJECT(ButterworthHighPass)
    ADD_OBJECT(ButterworthBandPass)
#if PY_MAJOR_VERSION == 3
	return m;
#endif
}

#endif /* __PYIIR1_MAIN__ */
