package layers;

import active.ActiveFunction;
import net.Net;
import utils.BlobUtil;
import utils.MathFunctionUtil;

public class ConvolutionLayer extends Layer {
	private Net net;
	private int layerId;
	/*
	 * 层类型
	 */
	private static final String layerType = "convolution";
	private Blob dataAndDiff;
	/*
	 * 卷积核(权重)，卷积核和偏置在开始时要随机初始化，这些参数是要在训练过程中学习的,kernel中保存data和diff(
	 * diff即Gradient)
	 */
	private Blob kernel;
	/*
	 * 偏置矩阵，卷积核和偏置在开始时要随机初始化，这些参数是要在训练过程中学习的,bias中保存data和diff(diff即Gradient)
	 */
	private Blob bias;
	/*
	 * 输入特征Map宽度
	 */
	private int width;
	/*
	 * 输入特征Map高度
	 */
	private int height;
	/*
	 * 输入特征Map数(不是必须的，当前层的输入Map即为上一层的输出Map，可以由前一层获取到，为了计算方便)
	 */
	private int inMapNums;
	/*
	 * 输出特征Map数,即卷积核（filter)的个数
	 */
	private int outMapNums;
	/*
	 * 卷积核大小,如果卷积核的长和宽不等，需要用kernel_h和kernel_w分别设定,这里假设长和宽相等
	 */
	private int kernelSize;
	/*
	 * 卷积扫描步长
	 */
	private int stride;
	/*
	 * 激活函数
	 */
	private ActiveFunction activeFunction;
	/*
	 * 以下参数未用到:https://blog.csdn.net/Missayaaa/article/details/81359438
	 * 学习率的系数，caffe中最终的学习率是这个数乘以solver.prototxt配置文件中的base_lr 如果有两个lr_mult,
	 * 则第一个表示权值的学习率，第二个表示偏置项的学习率。一般偏置项的学习率是权值学习率的两倍。
	 */
	@SuppressWarnings("unused")
	private float lr_mult;
	/*
	 * 扩充边缘，扩充的时候是左右、上下对称的，比如卷积核的大小为5*5，那么pad设置为2，则四个边缘都扩充2个像素，即宽度和高度都扩充了4个像素,
	 * 这样卷积运算之后的特征图就不会变小。也可以通过pad_h和pad_w来分别设定
	 */
	@SuppressWarnings("unused")
	private int pad;
	/*
	 * forward中产生的临时中间值，backword时需要用到，前提是有激活函数的情况下，用于临时保存前面的计算结果，用于激活函数的输入
	 */
	private Blob preActiveOutput;

	public ConvolutionLayer(Net net, int width, int height, int inMapNums, int outMapNums, int kernelSize, int stride) {
		this.net = net;
		this.width = width;
		this.height = height;
		this.inMapNums = inMapNums;
		this.outMapNums = outMapNums;
		this.kernelSize = kernelSize;
		this.stride = stride;
	}

	public void prepare() {
		// forward初始化卷积核和偏置
		if (kernel == null && bias == null) {
			kernel = new Blob(3, inMapNums * outMapNums, kernelSize, kernelSize);
			kernel.setData(new float[inMapNums * outMapNums * kernelSize * kernelSize]);
			bias = new Blob(1, outMapNums);
			bias.setData(new float[outMapNums]);
			// init params
			MathFunctionUtil.gaussianInitData(kernel.getData());
			MathFunctionUtil.constantInitData(bias.getData(), 0.001f);
		}
		preActiveOutput = new Blob(4, net.getBatchSize(), outMapNums, height, width);
		preActiveOutput.setData(new float[net.getBatchSize() * outMapNums * height * width]);
		// backforward计算梯度
		kernel.setDiff(new float[inMapNums * outMapNums * kernelSize * kernelSize]);
		bias.setDiff(new float[outMapNums]);
	}

	@Override
	public void initOutputDataAndDiff() {
		this.dataAndDiff = new Blob(4, net.getBatchSize(), outMapNums, height, width);
		this.dataAndDiff.setData(new float[net.getBatchSize() * outMapNums * height * width]);
		this.dataAndDiff.setDiff(new float[net.getBatchSize() * outMapNums * height * width]);

	}

	@Override
	public void forward(Blob preLayerDataBlob) {
		// TODO Auto-generated method stub
		Blob input = preLayerDataBlob;
		Blob output = this.dataAndDiff;
		float[] outputData = output.getData();
		float[] zData = this.preActiveOutput.getData();

		// 激活函数
		if (activeFunction != null) {
			// 卷积后的结果存贮在z中
			BlobUtil.fillValue(zData, 0);
			MathFunctionUtil.convBlobOperation(net, input, kernel, bias, this.preActiveOutput);
			for (int n = 0; n < output.getNum(); n++) {
				for (int c = 0; c < output.getChannels(); c++) {
					for (int h = 0; h < output.getHeight(); h++) {
						for (int w = 0; w < output.getWidth(); w++) {
							outputData[output.getIndexByParams(n, c, h, w)] = activeFunction
									.dataActive(zData[this.preActiveOutput.getIndexByParams(n, c, h, w)]);
						}
					}
				}
			}
		} else {
			// 卷积后的结果存贮在output中
			BlobUtil.fillValue(output.getData(), 0);
			MathFunctionUtil.convBlobOperation(net, input, kernel, bias, output);
		}

	}

	@Override
	public void backward(Blob preLayerDataAndDiffBlob) {
		// TODO Auto-generated method stub
		Blob input = preLayerDataAndDiffBlob;
		Blob inputDiff = this.dataAndDiff;
		Blob outputDiff = preLayerDataAndDiffBlob;
		float[] inputDiffData = inputDiff.getDiff();
		float[] zData = this.preActiveOutput.getData();
		float[] kernelGradientData = kernel.getDiff();
		float[] inputData = input.getData();
		float[] biasGradientData = bias.getDiff();

		// 先乘激活函数的导数,得到该层的误差
		if (activeFunction != null) {
			for (int n = 0; n < inputDiff.getNum(); n++) {
				for (int c = 0; c < inputDiff.getChannels(); c++) {
					for (int h = 0; h < inputDiff.getHeight(); h++) {
						for (int w = 0; w < inputDiff.getWidth(); w++) {
							inputDiffData[inputDiff.getIndexByParams(n, c, h, w)] *= activeFunction
									.diffActive(zData[this.preActiveOutput.getIndexByParams(n, c, h, w)]);
						}
					}
				}
			}
		}

		// 然后更新参数
		// 计算kernelGradient,这里并不更新kernel,kernel在优化器中更新
		BlobUtil.fillValue(kernel.getDiff(), 0);
		// workers.clear();
		for (int n = 0; n < inputDiff.getNum(); n++) {
			for (int ci = 0; ci < inputDiff.getChannels(); ci++) {
				for (int co = 0; co < outputDiff.getChannels(); co++) {
					for (int h = 0; h < inputDiff.getHeight(); h++) {
						for (int w = 0; w < inputDiff.getWidth(); w++) {
							// 先定位到输出的位置
							// 然后遍历kernel,通过kernel定位输入的位置
							// 然后将输入乘以diff
							int inStartX = w - kernel.getWidth() / 2;
							int inStartY = h - kernel.getHeight() / 2;
							// 和卷积核乘加

							for (int kh = 0; kh < kernel.getHeight(); kh++) {
								for (int kw = 0; kw < kernel.getWidth(); kw++) {
									int inY = inStartY + kh;
									int inX = inStartX + kw;
									if (inY >= 0 && inY < input.getHeight() && inX >= 0 && inX < input.getWidth()) {
										kernelGradientData[kernel.getIndexByParams(0,
												ci * outputDiff.getChannels() + co, kh,
												kw)] += inputData[input.getIndexByParams(n, co, inY, inX)]
														* inputDiffData[inputDiff.getIndexByParams(n, ci, h, w)];
									}
								}
							}
						}
					}
				}
			}
		}
		// 平均
		MathFunctionUtil.dataDivConstant(kernelGradientData, inputDiff.getNum());

		// 更新bias
		BlobUtil.fillValue(bias.getDiff(), 0);
		for (int n = 0; n < inputDiff.getNum(); n++) {
			for (int c = 0; c < inputDiff.getChannels(); c++) {
				for (int h = 0; h < inputDiff.getHeight(); h++) {
					for (int w = 0; w < inputDiff.getWidth(); w++) {
						biasGradientData[bias.getIndexByParams(0, 0, 0, c)] += inputDiffData[inputDiff
								.getIndexByParams(n, c, h, w)];
					}
				}
			}
		}
		// 平均
		MathFunctionUtil.dataDivConstant(biasGradientData, inputDiff.getNum());

		if (this.layerId <= 2)
			return;
		// 先把kernel旋转180度
		// Blob kernelRoate180 = MathFunctions.rotate180Blob(kernel);
		// 然后再做卷积
		BlobUtil.fillValue(outputDiff.getDiff(), 0);
		MathFunctionUtil.convBlobOperation(net, inputDiff, kernel, outputDiff);

		net.updateWeight(kernel);
		net.updateWeight(bias);
	}

	public ActiveFunction getActiveFunction() {
		return activeFunction;
	}

	public void setActiveFunction(ActiveFunction activeFunction) {
		this.activeFunction = activeFunction;
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
