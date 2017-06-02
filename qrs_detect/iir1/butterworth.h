#ifndef __BUTTERWORTH_H__
#define __BUTTERWORTH_H__

#include <Python.h>
#include <iir.h>
#include "config.h"

typedef struct {
    PyObject_HEAD
    pyiir1_HEAD
    /* Type-specific fields go here. */
    // Iir::Butterworth::LowPass<MAX_ORDER> * iir1;		/* Pointer to filter object */
} pyiir1_FilterButterworthLowPass;

typedef struct {
    PyObject_HEAD
    pyiir1_HEAD
    /* Type-specific fields go here. */
    // Iir::Butterworth::HighPass<MAX_ORDER> * iir1;        /* Pointer to filter object */
} pyiir1_FilterButterworthHighPass;

typedef struct {
    PyObject_HEAD
    pyiir1_HEAD
    /* Type-specific fields go here. */
    // Iir::Butterworth::BandPass<MAX_ORDER> * iir1;        /* Pointer to filter object */
} pyiir1_FilterButterworthBandPass;

extern PyTypeObject pyiir1_filterButterworthLowPassType;
extern PyTypeObject pyiir1_filterButterworthHighPassType;
extern PyTypeObject pyiir1_filterButterworthBandPassType;

#endif /* __BUTTERWORTH_H__ */