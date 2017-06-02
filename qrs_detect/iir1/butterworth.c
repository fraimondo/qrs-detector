#include "butterworth.h"
#include "config.h"


FUNC_DEALLOC(ButterworthLowPass);
FUNC_DEALLOC(ButterworthHighPass);
FUNC_DEALLOC(ButterworthBandPass);

FUNC_NEW(ButterworthLowPass);
FUNC_NEW(ButterworthHighPass);
FUNC_NEW(ButterworthBandPass);

DEFORD15(ButterworthLowPass, Iir::Butterworth::LowPass);
DEFORD15(ButterworthHighPass, Iir::Butterworth::HighPass);
DEFORD15(ButterworthBandPass, Iir::Butterworth::BandPass);

FUNC_INIT_DEF(ButterworthLowPass)
FUNC_INITKWARGS_LIST("lpass")
FUNC_INITWKARGS_PARSE("d", &lpass)
FUNC_INIT_DEBUG
eprintf("DEBUG:: lpass %f\n", lpass);
FUNC_INIT_ERROR(lpass <= 0, "lpass cannot be <= 0")
FUNC_INIT_INIT(ButterworthLowPass, Iir::Butterworth::LowPass, lpass)
FUNC_INIT_FINALIZE

FUNC_INIT_DEF(ButterworthHighPass)
FUNC_INITKWARGS_LIST("hpass")
FUNC_INITWKARGS_PARSE("d", &hpass)
FUNC_INIT_DEBUG
eprintf("DEBUG:: hpass %f\n", hpass);
FUNC_INIT_ERROR(hpass <= 0, "hpass cannot be <=0")
FUNC_INIT_INIT(ButterworthHighPass, Iir::Butterworth::HighPass, hpass)
FUNC_INIT_FINALIZE


FUNC_INIT_DEF(ButterworthBandPass)
FUNC_INITKWARGS_LIST("hpass", "lpass")
FUNC_INITWKARGS_PARSE("dd", &hpass, &lpass)
FUNC_INIT_DEBUG
eprintf("DEBUG:: hpass %f lpass %f\n", hpass, lpass);
FUNC_INIT_ERROR(lpass <= 0, "lpass cannot be <= 0")
FUNC_INIT_ERROR(lpass <= hpass, "lpass cannot be <= hpass")
FUNC_INIT_INIT(ButterworthBandPass, Iir::Butterworth::BandPass, (((lpass-hpass)/2)+hpass), ((lpass-hpass)))
FUNC_INIT_FINALIZE


FUNC_RESET(ButterworthLowPass);
FUNC_RESET(ButterworthHighPass);
FUNC_RESET(ButterworthBandPass);

FUNC_FILTER(ButterworthLowPass, self->lpass);
FUNC_FILTER(ButterworthHighPass, self->hpass);
FUNC_FILTER(ButterworthBandPass, (self->lpass-self->hpass)/2, (self->lpass-self->hpass));

FUNC_STR(ButterworthLowPass);
FUNC_STR(ButterworthHighPass);
FUNC_STR(ButterworthBandPass);

MEMBER_DEF(ButterworthLowPass);
MEMBER_DEF(ButterworthHighPass);
MEMBER_DEF(ButterworthBandPass);

METHOD_DEF(ButterworthLowPass);
METHOD_DEF(ButterworthHighPass);
METHOD_DEF(ButterworthBandPass);

OBJECT_DEF(ButterworthLowPass);
OBJECT_DEF(ButterworthHighPass);
OBJECT_DEF(ButterworthBandPass);