using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Tests
{
    [TestClass]
    public sealed class FuzzTests
    {
        // Number of vector dimensions. Larger values give better capacity but are slower.
        private const int Dims = 1024;

        // RNG seed used for both the memory and the triple generator so the test
        // is fully deterministic.
        private const int Seed = 7;

        // Pool sizes for the randomly-generated vocabulary.
        private const int EntityCount = 20;
        private const int PredicateCount = 5;

        // Total number of unique triples to insert across the whole test.
        private const int TotalFacts = 100;

        // Print an accuracy snapshot every N inserted facts.
        private const int CheckpointInterval = 10;

        // Accuracy threshold applied at the very first checkpoint (10 facts).
        // With only 10 facts in a 1 024-dimensional memory the retrieval rate
        // should be high; lower thresholds are expected as capacity fills up.
        private const double FirstCheckpointAccuracyThreshold = 0.70;

        public TestContext TestContext { get; set; } = null!;

        [TestMethod]
        public void Fuzz_RandomTriples_PrintsAccuracyStats()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);
            var rng = new Random(Seed);

            // ----------------------------------------------------------------
            // Build a fixed vocabulary of entities and predicates
            // ----------------------------------------------------------------
            var entities = Enumerable.Range(0, EntityCount)
                .Select(i => memory.CreateEntity($"Entity_{i}"))
                .ToArray();

            var predicates = Enumerable.Range(0, PredicateCount)
                .Select(i => memory.CreatePredicate($"Pred_{i}"))
                .ToArray();

            // ----------------------------------------------------------------
            // Generate unique (subject, predicate, object) triples
            // ----------------------------------------------------------------
            var facts = new List<(MemoryEntity<float> S, MemoryPredicate<float> P, MemoryEntity<float> O)>();
            var seen = new HashSet<(int, int, int)>();

            while (facts.Count < TotalFacts)
            {
                var si = rng.Next(EntityCount);
                var pi = rng.Next(PredicateCount);
                var oi = rng.Next(EntityCount);

                // Exclude reflexive triples and exact duplicates
                if (si == oi || !seen.Add((si, pi, oi)))
                    continue;

                facts.Add((entities[si], predicates[pi], entities[oi]));
            }

            // ----------------------------------------------------------------
            // Insert facts one-by-one, measuring accuracy at each checkpoint
            // ----------------------------------------------------------------
            var queryBuffer = new float[Dims];
            double firstCheckpointSubjectAccuracy = 0;

            TestContext.WriteLine("Facts | Subject acc | Object acc | Predicate acc");
            TestContext.WriteLine("------+-------------+------------+--------------");

            for (var i = 0; i < facts.Count; i++)
            {
                var (s, p, o) = facts[i];
                memory.Store(s, p, o);

                var storedCount = i + 1;
                if (storedCount % CheckpointInterval != 0 && storedCount != facts.Count)
                    continue;

                // Measure retrieval accuracy for every fact inserted so far
                var subjectHits = 0;
                var objectHits = 0;
                var predicateHits = 0;

                for (var f = 0; f < storedCount; f++)
                {
                    var (fs, fp, fo) = facts[f];

                    // Who is the subject? (query with predicate + object)
                    memory.QuerySubjects(fp, fo, queryBuffer);
                    var bestSubject = entities.MaxBy(e => Dot(queryBuffer, e.Vector.Span))!;
                    if (bestSubject.Equals(fs))
                        subjectHits++;

                    // What is the object? (query with subject + predicate)
                    memory.QueryObjects(fs, fp, queryBuffer);
                    var bestObject = entities.MaxBy(e => Dot(queryBuffer, e.Vector.Span))!;
                    if (bestObject.Equals(fo))
                        objectHits++;

                    // What is the predicate? (query with subject + object)
                    memory.QueryPredicates(fs, fo, queryBuffer);
                    var bestPredicate = predicates.MaxBy(pred => Dot(queryBuffer, pred.Vector.Span))!;
                    if (bestPredicate.Equals(fp))
                        predicateHits++;
                }

                var subjectAcc = (double)subjectHits / storedCount;
                var objectAcc = (double)objectHits / storedCount;
                var predicateAcc = (double)predicateHits / storedCount;

                TestContext.WriteLine(
                    $"{storedCount,5} | {subjectAcc,10:P1} | {objectAcc,9:P1} | {predicateAcc,12:P1}");

                if (storedCount == CheckpointInterval)
                    firstCheckpointSubjectAccuracy = subjectAcc;
            }

            // After only a handful of facts the memory should still recall most
            // subjects correctly – assert this to make the test genuinely fail on
            // regressions rather than just printing numbers.
            Assert.IsGreaterThan(
                FirstCheckpointAccuracyThreshold,
                firstCheckpointSubjectAccuracy,
                $"Subject retrieval accuracy at {CheckpointInterval} facts ({firstCheckpointSubjectAccuracy:P1}) " +
                $"should exceed {FirstCheckpointAccuracyThreshold:P0}");
        }
    }
}
