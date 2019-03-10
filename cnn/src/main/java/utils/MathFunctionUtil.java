package utils;

import java.util.Random;

import layers.Blob;
import net.Net;

public class MathFunctionUtil {
	public static void dataDivConstant(float[] data, float constant) {
		for (int i = 0; i < data.length; i++) {
			data[i] /= constant;
		}
	}

	/**
	 * 高斯分布初始化
	 * 
	 * @param data
	 */
	public static void gaussianInitData(float[] data) {
		Random random = new Random();
		for (int i = 0; i < data.length; i++) {
			data[i] = (float) (random.nextGaussian() * 0.1);
		}
	}

	/**
	 * 常量初始化
	 * 
	 * @param data
	 * @param value
	 */
	public static void constantInitData(float[] data, float value) {
		for (int i = 0; i < data.length; i++) {
			data[i] = value;
		}
	}

	/**
	 * 卷积操作，结果保存到output
	 * 
	 * @param net
	 * @param input
	 * @param kernel
	 * @param bias
	 * @param output
	 */
	public static void convBlobOperation(Net net, Blob input, Blob kernel, Blob bias, Blob output) {
		float[] inputData = input.getData();
		float[] kernelData = kernel.getData();
		float[] biasData = bias.getData();
		float[] outputData = output.getData();
		for (int n = 0; n < output.getNum(); n++) {
			for (int co = 0; co < output.getChannels(); co++) {
				for (int ci = 0; ci < input.getChannels(); ci++) {
					for (int h = 0; h < output.getHeight(); h++) {
						for (int w = 0; w < output.getWidth(); w++) {
							// 先定位到输出的位置
							// 然后遍历kernel,通过kernel定位输入的位置
							// 然后将输入乘以kernel
							int inStartX = w - kernel.getWidth() / 2;
							int inStartY = h - kernel.getHeight() / 2;
							// 和卷积核乘加
							for (int kh = 0; kh < kernel.getHeight(); kh++) {
								for (int kw = 0; kw < kernel.getWidth(); kw++) {
									int inY = inStartY + kh;
									int inX = inStartX + kw;
									if (inY >= 0 && inY < input.getHeight() && inX >= 0 && inX < input.getWidth()) {
										outputData[output.getIndexByParams(n, co, h,
												w)] += kernelData[kernel.getIndexByParams(0,
														co * input.getChannels() + ci, kh, kw)]
														* inputData[input.getIndexByParams(n, ci, inY, inX)];
									}
								}
							}

							// 加偏置
							if (bias != null) {
								outputData[output.getIndexByParams(n, co, h, w)] += biasData[bias.getIndexByParams(0, 0,
										0, co)];
							}
						}
					}
				}
			}
		}
	}

	public static void convBlobOperation(Net net, Blob input, Blob kernel, Blob output) {
		float[] inputData = input.getDiff();
		float[] kernelData = kernel.getData();
		float[] outputData = output.getDiff();
		for (int n = 0; n < input.getNum(); n++) {
			for (int ci = 0; ci < input.getChannels(); ci++) {
				for (int co = 0; co < output.getChannels(); co++) {
					for (int h = 0; h < input.getHeight(); h++) {
						for (int w = 0; w < input.getWidth(); w++) {

							int inStartX = w - kernel.getWidth() / 2;
							int inStartY = h - kernel.getHeight() / 2;
							// 和卷积核乘加
							for (int kh = 0; kh < kernel.getHeight(); kh++) {
								for (int kw = 0; kw < kernel.getWidth(); kw++) {
									int inY = inStartY + kh;
									int inX = inStartX + kw;
									if (inY >= 0 && inY < output.getHeight() && inX >= 0 && inX < output.getWidth()) {
										outputData[output.getIndexByParams(n, co, inY,
												inX)] += kernelData[kernel.getIndexByParams(0,
														ci * output.getChannels() + co, kh, kw)]
														* inputData[input.getIndexByParams(n, ci, h, w)];
									}
								}
							}
						}
					}
				}
			}
		}
	}

}
