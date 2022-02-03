# same imports as earlier.
from vtkmodules.vtkCommonDataModel import vtkDataSet
from vtkmodules.util.vtkAlgorithm import VTKPythonAlgorithmBase
from vtkmodules.numpy_interface import dataset_adapter as dsa

# new module for ParaView-specific decorators.
from paraview.util.vtkAlgorithm import smproxy, smproperty, smdomain

@smproxy.filter(label="Test Filter")
@smproperty.input(name="Input")
@smdomain.datatype(dataTypes=["vtkDataSet"], composite_data_supported=False)
class TestFilter(VTKPythonAlgorithmBase):
    def __init__(self):
        super().__init__(nInputPorts=1, nOutputPorts=1, outputType="vtkDataSet")
        self.lerp = 0.5

    def RequestDataObject(self, request, inInfo, outInfo):
        inData = self.GetInputData(inInfo, 0, 0)
        outData = self.GetOutputData(outInfo, 0)
        assert inData is not None
        if outData is None or (not outData.IsA(inData.GetClassName())):
            outData = inData.NewInstance()
            outInfo.GetInformationObject(0).Set(outData.DATA_OBJECT(), outData)
        return super().RequestDataObject(request, inInfo, outInfo)

    def RequestData(self, request, inInfo, outInfo):
        inData = self.GetInputData(inInfo, 0, 0)
        outData = self.GetOutputData(outInfo, 0)
        #print("input type =", inData.GetClassName())
        #print("output type =", outData.GetClassName())
        assert outData.IsA(inData.GetClassName())

        data = vtkDataSet.GetData(inInfo[0])

        pd = dsa.WrapDataObject(data).PointData
        b = pd['b']
        n = pd['n']
        sf = self.lerp*b+ (1-self.lerp)*n
        sf -= sf.min()
        sf /= sf.max()
        out_data = vtkDataSet.GetData(outInfo)

        output = dsa.WrapDataObject(out_data)

        dims = data.GetDimensions()
        output.SetDimensions(dims[0], dims[1], dims[2])

        #print(vtkDataSet.GetData(outInfo))

        output.PointData.append(sf, "sf")
        output.PointData.SetActiveScalars("sf")
        return 1

    @smproperty.doublevector(name="lerp", default_values=0.5)
    @smdomain.doublerange(min=0, max=1.0)
    def SetLerp(self, x):
        self.lerp = x
        self.Modified()
        self.Update()
    