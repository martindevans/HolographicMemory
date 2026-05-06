using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Tests
{
    [TestClass]
    public sealed class HolographicMemoryTests
    {
        // Use a moderate dimension that is large enough for reliable similarity
        // scores but small enough for fast tests.
        private const int Dims = 512;
        private const int Seed = 42;

        // Similarity threshold: stored facts should score above this, unrelated
        // facts should score below it.
        private const float HighSimilarity = 0.05f;
        private const float LowSimilarity = -0.05f;

        // ---------------------------------------------------------------------------
        // Vector generation
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void CreateSubject_IsDeterministic()
        {
            var m1 = new HolographicMemory<float>(Dims, Seed);
            var m2 = new HolographicMemory<float>(Dims, Seed);

            var a1 = m1.CreateSubject("Alice");
            var a2 = m2.CreateSubject("Alice");

            Assert.IsTrue(a1.Vector.Span.SequenceEqual(a2.Vector.Span));
        }

        [TestMethod]
        public void CreatePredicate_IsDeterministic()
        {
            var m1 = new HolographicMemory<float>(Dims, Seed);
            var m2 = new HolographicMemory<float>(Dims, Seed);

            var p1 = m1.CreatePredicate("Likes");
            var p2 = m2.CreatePredicate("Likes");

            Assert.IsTrue(p1.Vector.Span.SequenceEqual(p2.Vector.Span));
        }

        [TestMethod]
        public void CreateObject_IsDeterministic()
        {
            var m1 = new HolographicMemory<float>(Dims, Seed);
            var m2 = new HolographicMemory<float>(Dims, Seed);

            var o1 = m1.CreateObject("Pizza");
            var o2 = m2.CreateObject("Pizza");

            Assert.IsTrue(o1.Vector.Span.SequenceEqual(o2.Vector.Span));
        }

        [TestMethod]
        public void CreateProperty_IsDeterministic()
        {
            var m1 = new HolographicMemory<float>(Dims, Seed);
            var m2 = new HolographicMemory<float>(Dims, Seed);

            var p1 = m1.CreateProperty("Female");
            var p2 = m2.CreateProperty("Female");

            Assert.IsTrue(p1.Vector.Span.SequenceEqual(p2.Vector.Span));
        }

        [TestMethod]
        public void DifferentNames_ProduceDifferentVectors()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var bob = memory.CreateSubject("Bob");

            Assert.IsFalse(alice.Vector.Span.SequenceEqual(bob.Vector.Span));
        }

        // ---------------------------------------------------------------------------
        // Store + Query (subject-predicate-object)
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Query_PredicateObject_ReturnsStoredSubject()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var martin = memory.CreateSubject("Martin");
            var likes = memory.CreatePredicate("Likes");
            var anime = memory.CreateObject("Anime");

            memory.Store(alice, likes, anime);

            var result = new float[Dims];
            memory.Query(likes, anime, result);

            float aliceSim = Dot(result, alice.Vector.Span);
            float martinSim = Dot(result, martin.Vector.Span);

            Assert.IsTrue(aliceSim > HighSimilarity, $"Alice similarity {aliceSim} should be high");
            Assert.IsTrue(aliceSim > martinSim, $"Alice similarity {aliceSim} should exceed Martin's {martinSim}");
        }

        [TestMethod]
        public void Query_SubjectPredicate_ReturnsStoredObject()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateObject("Opera");
            var pizza = memory.CreateObject("Pizza");

            memory.Store(alice, likes, opera);

            var result = new float[Dims];
            memory.Query(alice, likes, result);

            float operaSim = Dot(result, opera.Vector.Span);
            float pizzaSim = Dot(result, pizza.Vector.Span);

            Assert.IsTrue(operaSim > HighSimilarity, $"Opera similarity {operaSim} should be high");
            Assert.IsTrue(operaSim > pizzaSim, $"Opera similarity {operaSim} should exceed Pizza's {pizzaSim}");
        }

        [TestMethod]
        public void Query_MultipleFacts_AreDiscriminated()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var martin = memory.CreateSubject("Martin");
            var likes = memory.CreatePredicate("Likes");
            var anime = memory.CreateObject("Anime");
            var opera = memory.CreateObject("Opera");

            memory.Store(martin, likes, anime);
            memory.Store(alice, likes, opera);

            var result = new float[Dims];

            // Query: who likes anime? → Martin
            memory.Query(likes, anime, result);
            float martinSim = Dot(result, martin.Vector.Span);
            float aliceSim = Dot(result, alice.Vector.Span);
            Assert.IsTrue(martinSim > aliceSim, $"Martin ({martinSim}) should score higher than Alice ({aliceSim}) for Anime");

            // Query: who likes opera? → Alice
            memory.Query(likes, opera, result);
            martinSim = Dot(result, martin.Vector.Span);
            aliceSim = Dot(result, alice.Vector.Span);
            Assert.IsTrue(aliceSim > martinSim, $"Alice ({aliceSim}) should score higher than Martin ({martinSim}) for Opera");
        }

        // ---------------------------------------------------------------------------
        // Store + Query (subject-property)
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Store_SubjectProperty_ChangesMemoryVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var female = memory.CreateProperty("Female");

            var normBefore = Norm(memory.MemoryVector);
            memory.Store(alice, female);
            var normAfter = Norm(memory.MemoryVector);

            Assert.IsTrue(normAfter > normBefore, $"Memory norm should increase after storing a property: before={normBefore}, after={normAfter}");
        }

        // ---------------------------------------------------------------------------
        // Remove
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Remove_Triple_DecreasesQuerySimilarity()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var martin = memory.CreateSubject("Martin");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateObject("Opera");
            var anime = memory.CreateObject("Anime");

            // Store two facts so the memory is non-empty after the remove
            memory.Store(alice, likes, opera);
            memory.Store(martin, likes, anime);

            var resultBefore = new float[Dims];
            memory.Query(likes, opera, resultBefore);
            float simBefore = Dot(resultBefore, alice.Vector.Span);

            memory.Remove(alice, likes, opera);

            var resultAfter = new float[Dims];
            memory.Query(likes, opera, resultAfter);
            float simAfter = Dot(resultAfter, alice.Vector.Span);

            Assert.IsTrue(simAfter < simBefore, $"Similarity after remove ({simAfter}) should be less than before ({simBefore})");
        }

        [TestMethod]
        public void Remove_Property_DecreasesMemoryVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var female = memory.CreateProperty("Female");

            memory.Store(alice, female);
            var beforeNorm = Norm(memory.MemoryVector);

            memory.Remove(alice, female);
            var afterNorm = Norm(memory.MemoryVector);

            Assert.IsTrue(afterNorm < beforeNorm, $"Memory norm should decrease after remove: before={beforeNorm}, after={afterNorm}");
        }

        // ---------------------------------------------------------------------------
        // Clear
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Clear_ZeroesMemoryVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateObject("Opera");
            memory.Store(alice, likes, opera);

            memory.Clear();

            foreach (var v in memory.MemoryVector)
                Assert.AreEqual(0f, v);
        }

        [TestMethod]
        public void Clear_ZeroesNormalizedMemoryVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateObject("Opera");
            memory.Store(alice, likes, opera);

            memory.Clear();

            foreach (var v in memory.NormalizedMemoryVector)
                Assert.AreEqual(0f, v);
        }

        // ---------------------------------------------------------------------------
        // NormalizedMemoryVector
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void NormalizedMemoryVector_HasUnitLength()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var likes = memory.CreatePredicate("Likes");
            var anime = memory.CreateObject("Anime");
            memory.Store(alice, likes, anime);

            float norm = Norm(memory.NormalizedMemoryVector);

            Assert.AreEqual(1f, norm, delta: 1e-5f);
        }

        [TestMethod]
        public void NormalizedMemoryVector_IsConsistentAfterMultipleAccesses()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var alice = memory.CreateSubject("Alice");
            var likes = memory.CreatePredicate("Likes");
            var anime = memory.CreateObject("Anime");
            memory.Store(alice, likes, anime);

            // Access twice; second access should not recompute and values should match
            var first = memory.NormalizedMemoryVector.ToArray();
            var second = memory.NormalizedMemoryVector.ToArray();

            CollectionAssert.AreEqual(first, second);
        }

        // ---------------------------------------------------------------------------
        // Dimension validation
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Store_WrongDimensions_Throws()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);
            var wrongVector = new float[Dims + 1];
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => memory.Store(wrongVector.AsSpan()));
        }

        [TestMethod]
        public void Remove_WrongDimensions_Throws()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);
            var wrongVector = new float[Dims - 1];
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => memory.Remove(wrongVector.AsSpan()));
        }

        // ---------------------------------------------------------------------------
        // CreateSubject/Predicate/Object/Property with explicit value
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void CreateSubject_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var customVec = new float[Dims];
            customVec[0] = 1f;

            var subject = memory.CreateSubject("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(subject.Vector.Span));
        }

        [TestMethod]
        public void CreatePredicate_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var customVec = new float[Dims];
            customVec[1] = 1f;

            var predicate = memory.CreatePredicate("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(predicate.Vector.Span));
        }

        [TestMethod]
        public void CreateObject_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var customVec = new float[Dims];
            customVec[2] = 1f;

            var obj = memory.CreateObject("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(obj.Vector.Span));
        }

        [TestMethod]
        public void CreateProperty_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicMemory<float>(Dims, Seed);

            var customVec = new float[Dims];
            customVec[3] = 1f;

            var prop = memory.CreateProperty("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(prop.Vector.Span));
        }
    }
}
