package layers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class Layer {
	public static void main(String[] args) {
		List<Integer> list = new ArrayList<>();
		for (int i = 0; i < 20; i++) {
			list.add(i);
		}
		for (int i = 0; i < list.size(); i++) {
			System.out.println(list.get(i));
		}
		Collections.shuffle(list);
		for (int i = 0; i < list.size(); i++) {
			System.out.println(list.get(i));
		}
	}

	abstract public void prepare();

	abstract public void initOutputDataAndDiff();

	abstract public void forward(Blob preLayerDataBlob);

	abstract public void backward(Blob preLayerDataAndDiffBlob);

	abstract public Blob getDataAndDiff();

}
