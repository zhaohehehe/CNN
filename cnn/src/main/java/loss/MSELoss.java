package loss;

import layers.Blob;
import utils.BlobUtil;

/**
 * MSE值越小，说明模型拟合实验数据的能力越强
 * http://baijiahao.baidu.com/s?id=1601092478839269810&wfr=spider&for=pc
 * 
 * @author zhaohe
 *
 */
public class MSELoss implements Loss {

	/**
	 * 代价函数的梯度，如果没有达到理想的拟合程度（还没有到达山底，继续寻找合适梯度下山）
	 * https://www.jianshu.com/p/c7e642877b0e
	 */
	@Override
	public void diff(Blob labelBlob, Blob outputDataAndDiffBlob) {
		float[] labelData = labelBlob.getData();
		float[] outputData = outputDataAndDiffBlob.getData();
		float[] diffData = outputDataAndDiffBlob.getDiff();
		int width = labelBlob.getWidth();
		int height = labelBlob.getHeight();
		float factor = 2;
		BlobUtil.fillValue(diffData, 0);
		for (int n = 0; n < height; n++) {
			for (int os = 0; os < width; os++) {
				diffData[n * width + os] += factor * (outputData[n * width + os] - labelData[n * width + os]);
			}
		}
	}

	/**
	 * 代价函数，预测值和实际值的Error = 偏差 + 方差，损失越小，拟合的越好
	 */
	@Override
	public float loss(Blob labelBlob, Blob outputDataBlob) throws Exception {
		float[] labelData = labelBlob.getData();
		float[] outputData = outputDataBlob.getData();
		float loss = 0.0f;
		for (int i = 0; i < labelBlob.getSize(); ++i) {
			loss += (labelData[i] - outputData[i]) * (labelData[i] - outputData[i]);
		}
		return loss / labelBlob.getHeight();
	}

}
