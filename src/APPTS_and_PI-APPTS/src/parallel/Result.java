package parallel;

public class Result {

    private int fitness_mfts;

    private int fitness_uncoveredCombs;

    public Result(int fitness_mft, int fitness_comb) {
        this.fitness_mfts = fitness_mft;
        this.fitness_uncoveredCombs = fitness_comb;
    }

    public static Result merge(Result r1, Result r2) {
        return new Result(r1.fitness_mfts + r2.fitness_mfts, r1.fitness_uncoveredCombs + r2.fitness_uncoveredCombs);
    }

    public int getFitness_mfts() {
        return fitness_mfts;
    }

    public int getFitness_uncoveredCombs() {
        return fitness_uncoveredCombs;
    }
}
