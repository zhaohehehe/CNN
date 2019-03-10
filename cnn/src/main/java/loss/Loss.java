package loss;

import layers.Blob;

public interface Loss {
	/**
	 * 输出(softmax)层损失，直到达到最小并保持平稳
	 * 
	 * @param labelBlob
	 *            输出(softmax)层 labelBlob
	 * @param outputDataAndDiffBlob
	 *            输出(softmax)层 DataAndDiffBlob
	 * @return
	 * @throws Exception
	 */
	public float loss(Blob labelBlob, Blob outputDataAndDiffBlob) throws Exception;

	/**
	 * 输出(softmax)层梯度
	 * 
	 * @param labelBlob
	 * @param outputDataAndDiffBlob
	 */
	public void diff(Blob labelBlob, Blob outputDataAndDiffBlob);
}
