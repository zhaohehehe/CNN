package network;

import java.io.IOException;
import java.util.List;

import dataset.DataSetElement;
import dataset.utils.DataSetUtil;
import net.Net;

public class NetWork {
	public static final String labelPath = "data/mnist/train-labels.idx1-ubyte";
	public static final String objPath = "data/mnist/train-images.idx3-ubyte";

	public static final String testLabelPath = "data/mnist/t10k-labels.idx1-ubyte";
	public static final String testObjPath = "data/mnist/t10k-images.idx3-ubyte";

	public static void main(String[] args) throws IOException {
		List<DataSetElement> tradinDataSet = DataSetUtil.readMnistDataSetUtil(labelPath, objPath);
		List<DataSetElement> testDataSet = DataSetUtil.readMnistDataSetUtil(testLabelPath, testObjPath);
		Net netInstance = new Net();
		netInstance.init(20);
		netInstance.buildCnnNet();
		netInstance.prepare();
		netInstance.train(tradinDataSet, 30, testDataSet);

		// netInstance.saveModel("model/mnist.model");
		// netInstance.loadModel("model/mnist.model");
		// netInstance.net.test(testDataSet);

	}

}
