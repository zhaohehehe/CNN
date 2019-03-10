package active;

public class ReluActivationFunc implements ActiveFunction {
	@Override
	public float dataActive(float in) {
		return Math.max(0, in);
	}

	@Override
	public float diffActive(float in) {
		// TODO Auto-generated method stub
		float result = in <= 0 ? 0.0f : 1.0f;
		return result;
	}

}
