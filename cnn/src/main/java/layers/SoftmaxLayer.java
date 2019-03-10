package layers;

import net.Net;
import utils.BlobUtil;

public class SoftmaxLayer extends Layer {
	private Net net;
	private int layerId;
	/*
	 * 层类型
	 */
	private static final String layerType = "softmax";
	private Blob dataAndDiff;
	private int classificationNums;

	public SoftmaxLayer(Net net, int classificationNums) {
		this.net = net;
		this.classificationNums = classificationNums;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub

	}

	public int getClassificationNums() {
		return classificationNums;
	}

	@Override
	public void initOutputDataAndDiff() {
		// 每张图片都会有一个分类向量，该图片对应的分类下标置1，其他置0
		this.dataAndDiff = new Blob(2, net.getBatchSize(), classificationNums);
		this.dataAndDiff.setData(new float[net.getBatchSize() * classificationNums]);
		this.dataAndDiff.setDiff(new float[net.getBatchSize() * classificationNums]);

	}

	@Override
	public void forward(Blob preLayerDataBlob) {
		// TODO Auto-generated method stub
		Blob input = preLayerDataBlob;
		Blob output = this.dataAndDiff;
		float[] inputData = input.getData();
		float[] outputData = output.getData();
		assert input.getSize() == output.getSize() : "SoftMax forward---- input.getSize()==output.getSize() error";

		for (int n = 0; n < net.getBatchSize(); n++) {
			float sum = 0.0f;
			float max = 0.001f;

			// 查找最大值
			for (int is = 0; is < input.getWidth(); is++) {
				max = Math.max(max, inputData[n * input.getWidth() + is]);
			}
			// 求和
			for (int is = 0; is < input.getWidth(); is++) {
				outputData[n * input.getWidth() + is] = (float) Math.exp(inputData[n * input.getWidth() + is] - max);
				sum += outputData[n * input.getWidth() + is];
			}
			if (sum == 0) {
				System.out.println("sum is zero");
				System.exit(0);
			}
			// 每一项除以sum
			for (int os = 0; os < output.getWidth(); os++) {
				outputData[n * output.getWidth() + os] /= sum;
			}

		}

	}

	@Override
	public void backward(Blob preLayerDataAndDiffBlob) {
		// TODO Auto-generated method stub
		Blob inputDiff = this.dataAndDiff;
		Blob outputDiff = preLayerDataAndDiffBlob;
		Blob output = this.dataAndDiff;
		float[] inputDiffData = inputDiff.getDiff();
		float[] outputDiffData = outputDiff.getDiff();
		float[] outputData = output.getData();
		assert inputDiff.getSize() == outputDiff
				.getSize() : "SoftMax backward---- inputDiff.getSize()==outputDiff.getSize() error";

		// 先求softmax函数的偏导数
		BlobUtil.fillValue(outputDiff.getDiff(), 0);
		for (int n = 0; n < net.getBatchSize(); n++) {
			for (int ods = 0; ods < outputDiff.getWidth(); ods++) {
				for (int ids = 0; ids < inputDiff.getWidth(); ids++) {
					if (ids == ods) {
						outputDiffData[n * output.getWidth() + ods] += outputData[n * output.getWidth() + ods]
								* (1.0 - outputData[n * output.getWidth() + ods])
								* inputDiffData[n * output.getWidth() + ids];
					} else {
						outputDiffData[n * output.getWidth() + ods] -= outputData[n * output.getWidth() + ods]
								* outputData[n * output.getWidth() + ids] * inputDiffData[n * output.getWidth() + ids];
					}
				}
			}
		}
	}

	public Blob getDataAndDiff() {
		return dataAndDiff;
	}

	public void setDataAndDiff(Blob dataAndDiff) {
		this.dataAndDiff = dataAndDiff;
	}

	public int getLayerId() {
		return layerId;
	}

	public void setLayerId(int layerId) {
		this.layerId = layerId;
	}

}
