package layers;

import net.Net;

public class InputDataLayer extends Layer {
	private Net net;
	private int layerId;
	/*
	 * 层类型
	 */
	private static final String layerType = "input";
	private Blob dataAndDiff;
	/*
	 * 输入特征Map宽度
	 */
	private int width;
	/*
	 * 输入特征Map高度
	 */
	private int height;
	/*
	 * 通道数,例如RGB通道数为3，黑白图片通道数为1
	 */
	private int channel;

	public InputDataLayer(Net net, int width, int height, int channel) {
		this.net = net;
		this.width = width;
		this.height = height;
		this.channel = channel;
	}

	@Override
	public void prepare() {
		// TODO Auto-generated method stub

	}
	public int getWidth() {
		return width;
	}

	public void setWidth(int width) {
		this.width = width;
	}

	public int getHeight() {
		return height;
	}

	public void setHeight(int height) {
		this.height = height;
	}

	public int getChannel() {
		return channel;
	}

	public void setChannel(int channel) {
		this.channel = channel;
	}

	public static String getLayertype() {
		return layerType;
	}

	@Override
	public void initOutputDataAndDiff() {
		this.dataAndDiff = new Blob(4, net.getBatchSize(), channel, height, width);
		this.dataAndDiff.setData(new float[net.getBatchSize() * channel * height * width]);
		this.dataAndDiff.setDiff(new float[net.getBatchSize() * channel * height * width]);

	}

	@Override
	public void forward(Blob preLayerDataBlob) {
		// TODO Auto-generated method stub

	}

	@Override
	public void backward(Blob preLayerDataAndDiffBlob) {
		// TODO Auto-generated method stub

	}

	@Override
	public Blob getDataAndDiff() {
		// TODO Auto-generated method stub
		return dataAndDiff;
	}

	public int getLayerId() {
		return layerId;
	}

	public void setLayerId(int layerId) {
		this.layerId = layerId;
	}

}
