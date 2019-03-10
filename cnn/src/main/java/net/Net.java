package net;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import active.ReluActivationFunc;
import dataset.DataSetElement;
import layers.Blob;
import layers.ConvolutionLayer;
import layers.FullConnectionLayer;
import layers.InputDataLayer;
import layers.Layer;
import layers.SoftmaxLayer;
import layers.SubSamplingLayer;
import loss.Loss;
import loss.MSELoss;
import optimizer.Optimizer;
import optimizer.SGDOptimizer;
import utils.BlobUtil;

public class Net {
	/*
	 * 网络的各层
	 */
	private List<Layer> layers = new ArrayList<>();
	/*
	 * 批量更新的大小
	 */
	private int batchSize = 1;
	/*
	 * 学习速录衰减，在对模型进行训练时，有可能遇到训练数据不够，即训练数据无法对整个数据的分布进行估计的时候，
	 * 或者在对模型进行过度训练（overtraining）时，常常会导致模型的过拟合（overfitting），即模型复杂度比实际数据复杂度还要高。
	 * 为了避免出现overfitting,会给误差函数添加一个惩罚项(下山步子不能太大，否则会错过最低点)
	 * https://blog.csdn.net/ztf312/article/details/51084950
	 */
	private float learningRateAttenuation = 0.8f;
	/*
	 * 损失函数，计算softmax层的误差和梯度
	 */
	private Loss loss;
	/*
	 * 每次反向训练利用SGD等算法优化除了softmax层以外的其他各层的bias和weight，直到softmax层最小化损失函数
	 */
	private Optimizer optimizer;

	public void init(int batchSize) {
		// 首先构建神经网络对象，并设置参数
		this.batchSize = batchSize;
		this.learningRateAttenuation = 0.9f;
		this.loss = new MSELoss();
		this.optimizer = new SGDOptimizer(0.1f);
	}

	public void buildCnnNet() {
		InputDataLayer inputDataLayer = new InputDataLayer(this, 28, 28, 1);
		inputDataLayer.setLayerId(1);
		this.layers.add(inputDataLayer);
		ConvolutionLayer convolutionLayer1 = new ConvolutionLayer(this, 28, 28, 1, 6, 3, 1);
		convolutionLayer1.setLayerId(2);
		convolutionLayer1.setActiveFunction(new ReluActivationFunc());
		this.layers.add(convolutionLayer1);
		SubSamplingLayer subSamplingLayer1 = new SubSamplingLayer(this, 28, 28, 6, 2, 2);
		subSamplingLayer1.setLayerId(3);
		this.layers.add(subSamplingLayer1);
		ConvolutionLayer convolutionLayer2 = new ConvolutionLayer(this, 14, 14, 6, 6, 3, 1);
		convolutionLayer2.setLayerId(4);
		convolutionLayer2.setActiveFunction(new ReluActivationFunc());
		this.layers.add(convolutionLayer2);
		SubSamplingLayer subSamplingLayer2 = new SubSamplingLayer(this, 14, 14, 6, 2, 2);
		subSamplingLayer2.setLayerId(5);
		this.layers.add(subSamplingLayer2);
		// 上一层池化层输出的特征map为7*7*6
		FullConnectionLayer fullConnectionLayer1 = new FullConnectionLayer(this, 7 * 7 * 6, 256);
		fullConnectionLayer1.setLayerId(6);
		fullConnectionLayer1.setActiveFunction(new ReluActivationFunc());
		this.layers.add(fullConnectionLayer1);
		FullConnectionLayer fullConnectionLayer2 = new FullConnectionLayer(this, 256, 10);
		fullConnectionLayer2.setLayerId(7);
		this.layers.add(fullConnectionLayer2);
		fullConnectionLayer2.setActiveFunction(new ReluActivationFunc());
		SoftmaxLayer softmaxLayer = new SoftmaxLayer(this, 10);
		softmaxLayer.setLayerId(8);
		this.layers.add(softmaxLayer);
	}

	public void prepare() {
		for (int i = 0; i < this.layers.size(); i++) {
			this.layers.get(i).initOutputDataAndDiff();
			this.layers.get(i).prepare();
		}
	}

	/**
	 * 
	 * @param trainDataSetList
	 * @param epoes
	 *            迭代次数
	 * @param batch
	 *            每次迭代批次
	 * @param testDataSetList
	 */
	public void train(List<DataSetElement> trainDataSetList, int epoes, List<DataSetElement> testDataSetList) {
		System.out.println("training...... please wait for a moment!");
		int batch = this.getBatchSize();
		float loclaLr = optimizer.getLr();
		float lossValue = 0;
		InputDataLayer input = (InputDataLayer) layers.get(0);
		for (int e = 0; e < epoes; e++) {
			Collections.shuffle(trainDataSetList);
			long start = System.currentTimeMillis();
			for (int i = 0; i <= trainDataSetList.size() - batch; i += batch) {
				List<Blob> inputAndLabel = fullInputDataSetAndOutputLabelBlobByDataSet(trainDataSetList, i, batch,
						input.getChannel(), input.getHeight(), input.getWidth());
				float tmpLoss = trainOnce(inputAndLabel.get(0), inputAndLabel.get(1));
				lossValue = (lossValue + tmpLoss) / 2;
				if ((i / batch) % 50 == 0) {
					System.out.println("==================" + i + " " + i + batch);
					System.out.println("==================" + lossValue + " " + tmpLoss);
					System.out.print(".");
				}
			}
			// 每个epoe做一次测试
			System.out.println();
			System.out.println("training...... epoe: " + e + " lossValue: " + lossValue + "  " + " lr: "
					+ optimizer.getLr() + "  " + " cost " + (System.currentTimeMillis() - start));

			test(testDataSetList);

			if (loclaLr > 0.0001f) {
				loclaLr *= this.learningRateAttenuation;
				optimizer.setLr(loclaLr);
			}
		}
	}

	public void test(List<DataSetElement> testDataSetList) {
		InputDataLayer inputDataLayer = (InputDataLayer) layers.get(0);
		System.out.println("testing...... please wait for a moment!");
		int correctCount = 0;
		int allCount = 0;
		int i = 0;
		for (i = 0; i <= testDataSetList.size() - batchSize; i += batchSize) {
			allCount += batchSize;
			List<Blob> inputAndLabel = fullInputDataSetAndOutputLabelBlobByDataSet(testDataSetList, i, batchSize,
					inputDataLayer.getChannel(), inputDataLayer.getHeight(), inputDataLayer.getWidth());
			Blob output = predict(inputAndLabel.get(0));
			int[] calOutLabels = getBatchOutputLabel(output.getData());
			int[] realLabels = getBatchOutputLabel(inputAndLabel.get(1).getData());
			for (int kk = 0; kk < calOutLabels.length; kk++) {
				if (calOutLabels[kk] == realLabels[kk]) {
					correctCount++;
				}
			}
		}

		float accuracy = correctCount / (float) allCount;
		System.out.println("test accuracy is " + accuracy + " correctCount " + correctCount + " allCount " + allCount);
	}

	private int[] getBatchOutputLabel(float[] data) {
		SoftmaxLayer softmaxLayer = (SoftmaxLayer) layers.get(layers.size() - 1);
		int[] outLabels = new int[softmaxLayer.getDataAndDiff().getHeight()];
		int outDataSize = softmaxLayer.getDataAndDiff().getWidth();
		for (int n = 0; n < outLabels.length; n++) {
			int maxIndex = 0;
			float maxValue = 0;
			for (int i = 0; i < outDataSize; i++) {
				if (maxValue < data[n * outDataSize + i]) {
					maxValue = data[n * outDataSize + i];
					maxIndex = i;
				}
			}
			outLabels[n] = maxIndex;
		}
		return outLabels;
	}

	public Blob predict(Blob inputData) {
		InputDataLayer inputDataLayer = (InputDataLayer) layers.get(0);
		SoftmaxLayer softmaxLayer = (SoftmaxLayer) layers.get(layers.size() - 1);
		float[] inputDataLayerDataBlob = inputDataLayer.getDataAndDiff().getData();
		for (int i = 0; i < inputData.getData().length; i++) {
			inputDataLayerDataBlob[i] = inputData.getData()[i];// 首先填充输入层InputDataLayer的data
		}
		// 前向传播
		forward();
		// 返回最后一层的数据
		return softmaxLayer.getDataAndDiff();
	}

	public float trainOnce(Blob inputData, Blob labelData) {
		InputDataLayer inputDataLayer = (InputDataLayer) layers.get(0);
		SoftmaxLayer softmaxLayer = (SoftmaxLayer) layers.get(layers.size() - 1);
		float lossValue = 0;
		float[] inputDataLayerDataBlob = inputDataLayer.getDataAndDiff().getData();
		for (int i = 0; i < inputData.getData().length; i++) {
			inputDataLayerDataBlob[i] = inputData.getData()[i];// 首先填充输入层InputDataLayer的data
		}
		forward();
		// softmax层代价
		try {
			lossValue = loss.loss(labelData, softmaxLayer.getDataAndDiff());
		} catch (Exception e) {
			e.printStackTrace();
		}
		// softmax层梯度
		loss.diff(labelData, softmaxLayer.getDataAndDiff());// 计算输出层(softmax层)diff

		backward();

		return lossValue;
	}

	private void backward() {
		for (int i = layers.size() - 1; i > 0; i--) {
			layers.get(i).backward(layers.get(i - 1).getDataAndDiff());
		}
	}

	public void forward() {
		for (int i = 1; i <= layers.size() - 1; i++) {
			layers.get(i).forward(layers.get(i - 1).getDataAndDiff());
		}
	}

	/**
	 * 填充训练数据集的输入和输出Blob
	 * 
	 * @param trainDataSetList
	 * @param start
	 * @param batchSize
	 * @param channels
	 * @param height
	 * @param width
	 * @return
	 */
	public List<Blob> fullInputDataSetAndOutputLabelBlobByDataSet(List<DataSetElement> trainDataSetList, int start,
			int batchSize, int channels, int height, int width) {
		Blob input = new Blob(3, batchSize, channels, height, width);
		input.setData(new float[batchSize * channels * height * width]);
		Blob label = new Blob(2, batchSize,
				this.getLayers().get(this.getLayers().size() - 1).getDataAndDiff().getWidth());
		label.setData(
				new float[batchSize * this.getLayers().get(this.getLayers().size() - 1).getDataAndDiff().getWidth()]);
		BlobUtil.fillValue(label.getData(), 0);
		float[] blobData = input.getData();
		float[] labelData = label.getData();
		for (int i = start; i < (batchSize + start); i++) {
			DataSetElement img = trainDataSetList.get(i);
			byte[] imgData = img.objData;
			assert img.objData.length == input.getSize() : "buildBlobByImageList -- blob size error";
			for (int j = 0; j < imgData.length; j++) {
				blobData[(i - start) * input.getSize() + j] = (imgData[j] & 0xff) / 128.0f - 1;// normalize
																								// and
																								// centerlize(-1,1)
			}
			int labelValue = img.label;
			for (int j = 0; j < label.getWidth(); j++) {
				if (j == labelValue) {
					labelData[(i - start) * label.getWidth() + j] = 1;
				}
			}
		}
		List<Blob> inputAndLabel = new ArrayList<Blob>();
		inputAndLabel.add(input);
		inputAndLabel.add(label);
		return inputAndLabel;
	}

	public int getBatchSize() {
		return batchSize;
	}

	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}

	public void updateWeight(Blob weight) {
		optimizer.updateWeight(weight);
	}

	public void updateBias(Blob bias) {
		optimizer.updateWeight(bias);
	}

	public List<Layer> getLayers() {
		return layers;
	}

	public void setLayers(List<Layer> layers) {
		this.layers = layers;
	}

	public float getLearningRateAttenuation() {
		return learningRateAttenuation;
	}

	public void setLearningRateAttenuation(float learningRateAttenuation) {
		this.learningRateAttenuation = learningRateAttenuation;
	}

	public Loss getLoss() {
		return loss;
	}

	public void setLoss(Loss loss) {
		this.loss = loss;
	}

	public Optimizer getOptimizer() {
		return optimizer;
	}

	public void setOptimizer(Optimizer optimizer) {
		this.optimizer = optimizer;
	}

}
