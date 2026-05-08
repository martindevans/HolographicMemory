using System.Diagnostics;
using HolographicMemory.Retrieval;
using HolographicMemory.Storage;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Tests
{
    [TestClass]
    public sealed class FuzzTests
    {
        // Number of vector dimensions. Larger values give better capacity but are slower.
        private const int Dims = 2048;

        // ID to use for memory.
        private static readonly Guid Id = new Guid(345235, 12, 141, 255, 128, 64, 32, 16, 8, 4, 2);

        // Pool sizes for the randomly-generated vocabulary.
        private const int EntityCount = 30;
        private const int PredicateCount = 10;
        private const int PropertyCount = 16;

        // Total number of unique facts to insert across the whole test.
        private const int TotalTriples = 500;
        private const int TotalPropertyFacts = 120;

        // Print an accuracy snapshot every N inserted facts.
        private const int CheckpointInterval = 10;

        // Accuracy threshold applied at the very first checkpoint (10 facts) for triple queries.
        // lower thresholds are expected as capacity fills up.
        private const double FirstCheckpointTripleAccuracyThreshold = 0.70;

        // Properties have a smaller vocabulary (PropertyCount) and multiple entities can
        // share the same property, which makes subject-by-property queries noisier.
        // The threshold is therefore lower than for triple queries.
        private const double FirstCheckpointPropertyAccuracyThreshold = 0.55;

        public TestContext TestContext { get; set; } = null!;

        [TestMethod]
        public void Fuzz_RandomTriples_PrintsAccuracyStats()
        {
            var memory = new HolographicStorage<float>(Dims, Id);
            var rng = new Random(Id.GetHashCode());

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

            while (facts.Count < TotalTriples)
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
                FirstCheckpointTripleAccuracyThreshold,
                firstCheckpointSubjectAccuracy,
                $"Subject retrieval accuracy at {CheckpointInterval} facts ({firstCheckpointSubjectAccuracy:P1}) " +
                $"should exceed {FirstCheckpointTripleAccuracyThreshold:P0}");
        }

        [TestMethod]
        public void Fuzz_RandomProperties_PrintsAccuracyStats()
        {
            var memory = new HolographicStorage<float>(Dims, Id);
            var rng = new Random(Id.GetHashCode());

            // ----------------------------------------------------------------
            // Build a fixed vocabulary of entities and properties
            // ----------------------------------------------------------------
            var entities = Enumerable.Range(0, EntityCount)
                .Select(i => memory.CreateEntity($"Entity_{i}"))
                .ToArray();

            var properties = Enumerable.Range(0, PropertyCount)
                .Select(i => memory.CreateProperty($"Prop_{i}"))
                .ToArray();

            // ----------------------------------------------------------------
            // Generate unique (subject, property) pairs
            // ----------------------------------------------------------------
            var facts = new List<(MemoryEntity<float> S, MemoryProperty<float> P)>();
            var seen = new HashSet<(int, int)>();

            while (facts.Count < TotalPropertyFacts)
            {
                var si = rng.Next(EntityCount);
                var pi = rng.Next(PropertyCount);

                if (!seen.Add((si, pi)))
                    continue;

                facts.Add((entities[si], properties[pi]));
            }

            // ----------------------------------------------------------------
            // Insert facts one-by-one, measuring accuracy at each checkpoint
            // ----------------------------------------------------------------
            var queryBuffer = new float[Dims];
            double firstCheckpointPropertyAccuracy = 0;

            TestContext.WriteLine("Facts | Property acc | Subject acc");
            TestContext.WriteLine("------+--------------+------------");

            for (var i = 0; i < facts.Count; i++)
            {
                var (s, p) = facts[i];
                memory.Store(s, p);

                var storedCount = i + 1;
                if (storedCount % CheckpointInterval != 0 && storedCount != facts.Count)
                    continue;

                // Measure retrieval accuracy for every fact inserted so far
                var propertyHits = 0;
                var subjectHits = 0;

                for (var f = 0; f < storedCount; f++)
                {
                    var (fs, fp) = facts[f];

                    // What property does this entity have? (query with subject)
                    memory.QueryProperties(fs, queryBuffer);
                    var bestProperty = properties.MaxBy(prop => Dot(queryBuffer, prop.Vector.Span))!;
                    if (bestProperty.Equals(fp))
                        propertyHits++;

                    // Which entity has this property? (query with property)
                    memory.QuerySubjects(fp, queryBuffer);
                    var bestSubject = entities.MaxBy(e => Dot(queryBuffer, e.Vector.Span))!;
                    if (bestSubject.Equals(fs))
                        subjectHits++;
                }

                var propertyAcc = (double)propertyHits / storedCount;
                var subjectAcc = (double)subjectHits / storedCount;

                TestContext.WriteLine(
                    $"{storedCount,5} | {propertyAcc,11:P1} | {subjectAcc,10:P1}");

                if (storedCount == CheckpointInterval)
                    firstCheckpointPropertyAccuracy = propertyAcc;
            }

            // Assert that property retrieval accuracy at the first checkpoint is high.
            Assert.IsGreaterThan(
                FirstCheckpointPropertyAccuracyThreshold,
                firstCheckpointPropertyAccuracy,
                $"Property retrieval accuracy at {CheckpointInterval} facts ({firstCheckpointPropertyAccuracy:P1}) " +
                $"should exceed {FirstCheckpointPropertyAccuracyThreshold:P0}");
        }

        [TestMethod]
        [TestCategory("NonCI")]
        public void Fuzz_Retrieval()
        {
            TestContext.WriteLine("Facts | Subject acc | Object acc | Predicate acc | Milliseconds");
            TestContext.WriteLine("------+-------------+------------+---------------+-------------");

            Fuzz_Retrieval_Single(10);
            Fuzz_Retrieval_Single(100);
            Fuzz_Retrieval_Single(200);
            Fuzz_Retrieval_Single(300);
        }

        private void Fuzz_Retrieval_Single(int factCount)
        {
            var memory = new HolographicStorage<float>(Dims, Id);
            var rng = new Random(unchecked(Id.GetHashCode() * factCount));

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

            while (facts.Count < factCount)
            {
                var si = rng.Next(EntityCount);
                var pi = rng.Next(PredicateCount);
                var oi = rng.Next(EntityCount);

                // Exclude reflexive triples and exact duplicates
                if (si == oi || !seen.Add((si, pi, oi)))
                    continue;

                // Generate facts and insert into memory
                var (s, p, o) = (entities[si], predicates[pi], entities[oi]);
                facts.Add((s, p, o));
                memory.Store(s, p, o);
            }

            // ----------------------------------------------------------------
            // Now test retrieval
            // ----------------------------------------------------------------

            var retrieval = new HolographicRetrieval<float>(new TestVectorStorage(facts));

            var timer = new Stopwatch();
            timer.Start();
            
            var queryBuffer = new float[Dims];
            float subjectHits = 0;
            float objectHits = 0;
            float predicateHits = 0;
            foreach (var (fs, fp, fo) in facts)
            {
                // Who is the subject? (query with predicate + object)
                memory.QuerySubjects(fp, fo, queryBuffer);

                // Retrieve some results, take the correct answer if it's there
                var subjectResults = retrieval.Retrieve(memory.Id, MemoryVectorType.Entity, queryBuffer, 4).ToArray();
                var subjectResult = subjectResults.FirstOrDefault(a => a.Vector.VectorId == fs.Name);
                subjectHits += Convert.ToInt32(subjectResult != default);

                // What is the object? (query with subject + predicate)
                memory.QueryObjects(fs, fp, queryBuffer);

                // Retrieve some results, take the correct answer if it's there
                var objectResults = retrieval.Retrieve(memory.Id, MemoryVectorType.Entity, queryBuffer, 4).ToArray();
                var objectResult = objectResults.SingleOrDefault(a => a.Vector.VectorId == fo.Name);
                objectHits += Convert.ToInt32(objectResult != default);

                // What is the predicate? (query with subject + object)
                memory.QueryPredicates(fs, fo, queryBuffer);

                // Retrieve some results, take the correct answer if it's there
                var predicateResults = retrieval.Retrieve(memory.Id, MemoryVectorType.Predicate, queryBuffer, 4).ToArray();
                var predicateResult = predicateResults.FirstOrDefault(a => a.Vector.VectorId == fp.Name);
                predicateHits += Convert.ToInt32(predicateResult != default);
            }

            var subjectAcc = (double)subjectHits / facts.Count;
            var objectAcc = (double)objectHits / facts.Count;
            var predicateAcc = (double)predicateHits / facts.Count;
            var elapsed = timer.Elapsed;

            TestContext.WriteLine(
                $"{facts.Count,5} | {subjectAcc,10:P1} | {objectAcc,9:P1} | {predicateAcc,12:P1} | {elapsed.TotalMilliseconds}ms");
        }
    }

    internal class TestVectorStorage
        : IVectorStorage<float>
    {
        private readonly List<(MemoryEntity<float> S, MemoryPredicate<float> P, MemoryEntity<float> O)> _facts;

        public TestVectorStorage(List<(MemoryEntity<float> S, MemoryPredicate<float> P, MemoryEntity<float> O)> facts)
        {
            _facts = facts;
        }

        public IEnumerable<(float Similarity, RetrievalVector<float> Vector)> Search(Guid memory, MemoryVectorType type, ReadOnlyMemory<float> query, int max)
        {
            return (
                from triple in _facts.AsParallel()
                from item in new[] { Filter(triple.S), Filter(triple.P), Filter(triple.O) }
                orderby item.Item1 descending
                select item
            ).Take(max);

            (float, RetrievalVector<float>) Filter<TVector>(TVector vector) where TVector : BaseMemoryVector<TVector, float>
            {
                var sim = CosineSimilarity(vector.Vector.Span, query.Span);
                return (
                    sim,
                    new RetrievalVector<float>(memory, vector.Name, vector.Type, vector.Vector)
                );
            }
        }

        public (float Similarity, RetrievalVector<float> Vector)? Search(Guid memory, MemoryVectorType type, ReadOnlyMemory<float> vector)
        {
            return Search(memory, type, vector, 1).SingleOrDefault();
        }
    }
}
