package layers;

import net.Net;

public class SubSamplingLayer extends Layer {
	private Net net;
	private int layerId;
	/*
	 * 层类型
	 */
	private static final String layerType = "subsampling";
	private Blob dataAndDiff;
	/*
	 * 池化方法，默认为MAX。目前可用的方法有MAX, AVE, 重叠池化等
	 */
	private static final String poolMeythod = "MEAN";
	/*
	 * 输入特征Map宽度
	 */
	private int inWidth;
	/*
	 * 输入特征Map高度
	 */
	private int inHeight;
	/*
	 * 输入特征Map数
	 */
	private int inMapNums;
	/*
	 * 池化的核大小。也可以用kernel_h和kernel_w分别设定。
	 */
	private int kernelSize;
	/*
	 * 池化的步长，一般我们设置为kernelSize=stride，即不重叠。也可以用stride_h和stride_w来设置
	 */
	private int stride;
	/*
	 * 输出特征Map宽度,根据池化核大小、输入特征Map的高度和宽度以及池化方法可计算出来
	 */
	@SuppressWarnings("unused")
	private int outWidth;
	/*
	 * 输出特征Map高度，根据池化核大小、输入特征Map的高度和宽度以及池化方法可计算出来
	 */
	@SuppressWarnings("unused")
	private int outWheight;
	/*
	 * 和卷积层的pad的一样，进行边缘扩充
	 */
	@SuppressWarnings("unused")
	private float pad;
	/*
	 * forward中产生的临时中间值，backword时需要用到，前提是：池化方法是最大池化时用到
	 */
	private Blob maxIndexBlob;

	public SubSamplingLayer(Net net, int inWidth, int inHeight, int inMapNums, int kernelSize, int stride) {
		this.net = net;
		this.inWidth = inWidth;
		this.inHeight = inHeight;
		this.inMapNums = inMapNums;
		this.kernelSize = kernelSize;
		this.stride = stride;
	}

	@Override
	public void prepare() {
		if (poolMeythod.equals("MAX")) {
			maxIndexBlob = new Blob(4, net.getBatchSize(), inMapNums, inHeight, inWidth);
			maxIndexBlob.setData(new float[net.getBatchSize() * inMapNums * inHeight * inWidth]);
		}
		// TODO Auto-generated method stub

	}

	@Override
	public void initOutputDataAndDiff() {
		this.dataAndDiff = new Blob(4, net.getBatchSize(), inMapNums, inHeight / kernelSize, inWidth / kernelSize);
		this.dataAndDiff
				.setData(new float[net.getBatchSize() * inMapNums * inHeight / kernelSize * inWidth / kernelSize]);
		this.dataAndDiff
				.setDiff(new float[net.getBatchSize() * inMapNums * inHeight / kernelSize * inWidth / kernelSize]);

	}

	@Override
	public void forward(Blob preLayerDataBlob) {
		// TODO Auto-generated method stub
		Blob input = preLayerDataBlob;
		Blob output = this.dataAndDiff;
		float[] outputData = output.getData();
		float[] inputData = input.getData();
		for (int n = 0; n < output.getNum(); n++) {
			for (int c = 0; c < output.getChannels(); c++) {
				for (int h = 0; h < output.getHeight(); h++) {
					for (int w = 0; w < output.getWidth(); w++) {
						int inStartX = w * stride;
						int inStartY = h * stride;
						float sum = 0;
						for (int kh = 0; kh < kernelSize; kh++) {
							for (int kw = 0; kw < kernelSize; kw++) {
								int curIndex = input.getIndexByParams(n, c, inStartY + kh, inStartX + kw);
								sum += inputData[curIndex];
							}
						}
						outputData[output.getIndexByParams(n, c, h, w)] = sum / (kernelSize * kernelSize);
					}
				}
			}
		}
	}

	@Override
	public void backward(Blob preLayerDataAndDiffBlob) {
		// TODO Auto-generated method stub
		Blob inputDiff = this.dataAndDiff;
		Blob outputDiff = preLayerDataAndDiffBlob;
		float[] inputDiffData = inputDiff.getDiff();
		float[] outputDiffData = outputDiff.getDiff();
		for (int n = 0; n < inputDiff.getNum(); n++) {
			for (int c = 0; c < inputDiff.getChannels(); c++) {
				for (int h = 0; h < inputDiff.getHeight(); h++) {
					for (int w = 0; w < inputDiff.getWidth(); w++) {
						int inStartX = w * stride;
						int inStartY = h * stride;
						for (int kh = 0; kh < kernelSize; kh++) {
							for (int kw = 0; kw < kernelSize; kw++) {
								int curIndex = outputDiff.getIndexByParams(n, c, inStartY + kh, inStartX + kw);
								outputDiffData[curIndex] = inputDiffData[inputDiff.getIndexByParams(n, c, h, w)];
							}
						}

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
