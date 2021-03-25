import tensorflow

def isInnerBaseType(val):
    #return isinstance(val, int) or isinstance(val, list) or isinstance(val, dict) or isinstance(val, float) or isinstance(val, str)
    retval = True;
    retval = (isinstance(val, int) or isinstance(val, float) or isinstance(val, str) 
            or isinstance(val, type(tensorflow.float16)) or isinstance(val, type(tensorflow.float32)) or isinstance(val, type(tensorflow.float64))
            or isinstance(val, type(tensorflow.bfloat16)) or isinstance(val, type(tensorflow.complex64)) or isinstance(val, type(tensorflow.complex128)) or isinstance(val, type(tensorflow.int8))
            or isinstance(val, type(tensorflow.uint8)) or isinstance(val, type(tensorflow.uint16)) or isinstance(val, type(tensorflow.uint32)) or isinstance(val, type(tensorflow.uint64)) or isinstance(val, type(tensorflow.int16))
            or isinstance(val, type(tensorflow.int32)) or isinstance(val, type(tensorflow.int64)) or isinstance(val, type(tensorflow.bool)) or isinstance(val, type(tensorflow.string)) or isinstance(val, type(tensorflow.qint8))
            or isinstance(val, type(tensorflow.quint8)) or isinstance(val, type(tensorflow.qint16)) or isinstance(val, type(tensorflow.quint16)) or isinstance(val, type(tensorflow.qint32))
            or isinstance(val, type(tensorflow.resource)) or isinstance(val, type(tensorflow.variant)))
    return retval

def printHelpInfo(specname, levelnum ):
    #if specname.find("denominator")>0:
    #    return
    #if specname.find("imag")>0:
    #    return
    #if specname.find("numerator")>0:
    #    return
    #if specname.find("real")>0:
    #    return
    varnames = []
    try:
        varnames = eval("dir(%s)"%(specname))
    except :
        pass
    for key in varnames:
        #print(key.find("_"))
        if key.find("_")==0:
            continue
        tempname = "%s.%s"%(specname,key)
        isInner = False
        try:
            isInner = isInnerBaseType(eval(tempname))
        except:
            continue
        if isInner:
            continue
        #print(tempname)
        if eval("hasattr(%s, '__doc__')"%(tempname)):
            print("help(%s):\n%s\n\n"%(tempname, eval("%s.__doc__"%(tempname))))
        if levelnum<3:
            printHelpInfo(tempname, levelnum+1)

printHelpInfo('tensorflow', 0)



