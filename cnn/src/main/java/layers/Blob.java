package layers;

/**
 * Blob是Caffe的基本数据结构，具有CPU和GPU之间同步的能力,它是4维的数组(Num, Channels, Height, Width);
 * 设Blob数据维度为 number N x channel K x height H x width W，
 * Blob是row-major保存的，因此在(n, k, h, w)位置的值物理位置为((n * K + k) * H + h) * W +
 * w，其中Number/N是batch size。 Blob对象里面有2个属性，分别是一个四维数组
 * 数组的维度是：维度是:(num，channels，height，width)
 * 
 * @author zhaohe
 *
 */
public class Blob {

	// 输出data,网络间正向传的数据，如图像像素数据之类的
	private float[] data;
	// 反向gradient，Back propagation进行学习时，运算得到的梯度数据
	private float[] diff;

	private int num;
	private int channels;
	private int height;
	private int width;

	/*
	 * 临时存储，计算有用到
	 */
	private int dimention;

	public Blob(int dimension, int num, int channels, int height, int width) {
		super();
		this.num = num;
		this.channels = channels;
		this.height = height;
		this.width = width;
		this.dimention = dimension;
	}

	public Blob(int dimension, int channels, int height, int width) {
		super();
		this.channels = channels;
		this.height = height;
		this.width = width;
		this.dimention = dimension;
	}

	public Blob(int dimension, int height, int width) {
		super();
		this.height = height;
		this.width = width;
		this.dimention = dimension;
	}

	public Blob(int dimension, int width) {
		super();
		this.width = width;
		this.dimention = dimension;
	}

	public void setData(float[] data) {
		this.data = data;
	}

	public void setDiff(float[] diff) {
		this.diff = diff;
	}

	public float[] getData() {
		return data;
	}

	public float[] getDiff() {
		return diff;
	}

	public int getNum() {
		return num;
	}

	public void setNum(int num) {
		this.num = num;
	}

	public int getChannels() {
		return channels;
	}

	public void setChannels(int channels) {
		this.channels = channels;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public int getSize() {
		if (dimention == 4) {
			return this.num * this.channels * this.height * this.width;
		} else if (dimention == 3) {
			return this.channels * this.height * this.width;
		} else if (dimention == 2) {
			return this.height * this.width;
		} else {
			return this.width;
		}
	}

	public int getIndexByParams(int numbers, int channels, int height, int width) {
		int numIndex = this.channels * this.height * this.width;
		int channelIndex = this.height * this.width;
		int heightIndex = this.width;
		return (numbers * numIndex + channels * channelIndex + height * heightIndex + width);
	}

}
