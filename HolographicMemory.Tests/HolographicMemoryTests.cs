using HolographicMemory.Storage;
using static System.Numerics.Tensors.TensorPrimitives;

namespace HolographicMemory.Tests
{
    [TestClass]
    public sealed class HolographicMemoryTests
    {
        // Use a moderate dimension that is large enough for reliable similarity
        // scores but small enough for fast tests.
        private const int Dims = 1024;
        private static readonly Guid Id = new(345235, 12, 141, 255, 128, 64, 32, 16, 8, 4, 2);

        // Similarity threshold: stored facts should score above this, unrelated
        // facts should score below it.
        private const float HighSimilarity = 0.25f;

        // ---------------------------------------------------------------------------
        // Vector generation
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void CreateSubject_IsDeterministic()
        {
            var m1 = new HolographicStorage(Dims, Id);
            var m2 = new HolographicStorage(Dims, Id);

            var a1 = m1.CreateEntity("Alice");
            var a2 = m2.CreateEntity("Alice");

            Assert.IsTrue(a1.Vector.Span.SequenceEqual(a2.Vector.Span));
        }

        [TestMethod]
        public void CreatePredicate_IsDeterministic()
        {
            var m1 = new HolographicStorage(Dims, Id);
            var m2 = new HolographicStorage(Dims, Id);

            var p1 = m1.CreatePredicate("Likes");
            var p2 = m2.CreatePredicate("Likes");

            Assert.IsTrue(p1.Vector.Span.SequenceEqual(p2.Vector.Span));
        }

        [TestMethod]
        public void CreateObject_IsDeterministic()
        {
            var m1 = new HolographicStorage(Dims, Id);
            var m2 = new HolographicStorage(Dims, Id);

            var o1 = m1.CreateEntity("Pizza");
            var o2 = m2.CreateEntity("Pizza");

            Assert.IsTrue(o1.Vector.Span.SequenceEqual(o2.Vector.Span));
        }

        [TestMethod]
        public void CreateProperty_IsDeterministic()
        {
            var m1 = new HolographicStorage(Dims, Id);
            var m2 = new HolographicStorage(Dims, Id);

            var p1 = m1.CreateProperty("Female");
            var p2 = m2.CreateProperty("Female");

            Assert.IsTrue(p1.Vector.Span.SequenceEqual(p2.Vector.Span));
        }

        [TestMethod]
        public void DifferentNames_ProduceDifferentVectors()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var bob = memory.CreateEntity("Bob");

            Assert.IsFalse(alice.Vector.Span.SequenceEqual(bob.Vector.Span));
        }

        // ---------------------------------------------------------------------------
        // Store + Query (subject-predicate-object)
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Query_PredicateObject_ReturnsStoredSubject()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var martin = memory.CreateEntity("Martin");
            var likes = memory.CreatePredicate("Likes");
            var anime = memory.CreateEntity("Anime");

            memory.Store(alice, likes, anime);

            var result = new float[Dims];
            memory.QuerySubjects(likes, anime, result);

            var aliceSim = Dot(result, alice.Vector.Span);
            var martinSim = Dot(result, martin.Vector.Span);

            Assert.IsGreaterThan(HighSimilarity, aliceSim, $"Alice similarity {aliceSim} should be high");
            Assert.IsGreaterThan(martinSim, aliceSim, $"Alice similarity {aliceSim} should exceed Martin's {martinSim}");
        }

        [TestMethod]
        public void Query_SubjectPredicate_ReturnsStoredObject()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateEntity("Opera");
            var pizza = memory.CreateEntity("Pizza");

            memory.Store(alice, likes, opera);

            var result = new float[Dims];
            memory.QueryObjects(alice, likes, result);

            var operaSim = Dot(result, opera.Vector.Span);
            var pizzaSim = Dot(result, pizza.Vector.Span);

            Assert.IsGreaterThan(HighSimilarity, operaSim, $"Opera similarity {operaSim} should be high");
            Assert.IsGreaterThan(pizzaSim, operaSim, $"Opera similarity {operaSim} should exceed Pizza's {pizzaSim}");
        }

        [TestMethod]
        public void Query_SubjectObject_ReturnsStoredPredicate()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var likes = memory.CreatePredicate("Likes");
            var eats = memory.CreatePredicate("Eats");
            var opera = memory.CreateEntity("Opera");

            memory.Store(alice, likes, opera);

            var result = new float[Dims];
            memory.QueryPredicates(alice, opera, result);

            var likesSim = Dot(result, likes.Vector.Span);
            var eatsSim = Dot(result, eats.Vector.Span);

            Assert.IsGreaterThan(HighSimilarity, likesSim, $"Likes similarity {likesSim} should be high");
            Assert.IsGreaterThan(eatsSim, likesSim, $"Likes similarity {likesSim} should exceed Eats' {eatsSim}");
        }

        [TestMethod]
        public void Query_MultipleFacts_AreDiscriminated()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var martin = memory.CreateEntity("Martin");
            var likes = memory.CreatePredicate("Likes");
            var anime = memory.CreateEntity("Anime");
            var opera = memory.CreateEntity("Opera");

            memory.Store(martin, likes, anime);
            memory.Store(alice, likes, opera);

            var result = new float[Dims];

            // Query: who likes anime? → Martin
            memory.QuerySubjects(likes, anime, result);
            var martinSim = Dot(result, martin.Vector.Span);
            var aliceSim = Dot(result, alice.Vector.Span);
            Assert.IsGreaterThan(aliceSim, martinSim, $"Martin ({martinSim}) should score higher than Alice ({aliceSim}) for Anime");

            // Query: who likes opera? → Alice
            memory.QuerySubjects(likes, opera, result);
            martinSim = Dot(result, martin.Vector.Span);
            aliceSim = Dot(result, alice.Vector.Span);
            Assert.IsGreaterThan(martinSim, aliceSim, $"Alice ({aliceSim}) should score higher than Martin ({martinSim}) for Opera");
        }

        // ---------------------------------------------------------------------------
        // Store + Query (subject-property)
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Store_SubjectProperty_ChangesMemoryVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var female = memory.CreateProperty("Female");

            var normBefore = Norm(memory.MemoryVector);
            memory.Store(alice, female);
            var normAfter = Norm(memory.MemoryVector);

            Assert.IsGreaterThan(normBefore, normAfter, $"Memory norm should increase after storing a property: before={normBefore}, after={normAfter}");
        }

        [TestMethod]
        public void Query_Subject_ReturnsStoredProperty()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var woman = memory.CreateProperty("Woman");
            var man = memory.CreateProperty("Man");

            memory.Store(alice, woman);

            var result = new float[Dims];
            memory.QueryProperties(alice, result);

            var womanSim = Dot(result, woman.Vector.Span);
            var manSim = Dot(result, man.Vector.Span);

            Assert.IsGreaterThan(HighSimilarity, womanSim, $"Woman similarity {womanSim} should be high");
            Assert.IsGreaterThan(manSim, womanSim, $"Woman similarity {womanSim} should exceed Man's {manSim}");
        }

        [TestMethod]
        public void Query_Property_ReturnsMatchingSubject()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var martin = memory.CreateEntity("Martin");
            var woman = memory.CreateProperty("Woman");

            memory.Store(alice, woman);

            var result = new float[Dims];
            memory.QuerySubjects(woman, result);

            var aliceSim = Dot(result, alice.Vector.Span);
            var martinSim = Dot(result, martin.Vector.Span);

            Assert.IsGreaterThan(HighSimilarity, aliceSim, $"Alice similarity {aliceSim} should be high");
            Assert.IsGreaterThan(martinSim, aliceSim, $"Alice similarity {aliceSim} should exceed Martin's {martinSim}");
        }

        // ---------------------------------------------------------------------------
        // Remove
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Remove_Triple_DecreasesQuerySimilarity()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var martin = memory.CreateEntity("Martin");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateEntity("Opera");
            var anime = memory.CreateEntity("Anime");

            // Store two facts so the memory is non-empty after the remove
            memory.Store(alice, likes, opera);
            memory.Store(martin, likes, anime);

            var resultBefore = new float[Dims];
            memory.QuerySubjects(likes, opera, resultBefore);
            var simBefore = Dot(resultBefore, alice.Vector.Span);

            memory.Remove(alice, likes, opera);

            var resultAfter = new float[Dims];
            memory.QuerySubjects(likes, opera, resultAfter);
            var simAfter = Dot(resultAfter, alice.Vector.Span);

            Assert.IsLessThan(simBefore, simAfter, $"Similarity after remove ({simAfter}) should be less than before ({simBefore})");
        }

        [TestMethod]
        public void Remove_Property_DecreasesMemoryVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var female = memory.CreateProperty("Female");

            memory.Store(alice, female);
            var beforeNorm = Norm(memory.MemoryVector);

            memory.Remove(alice, female);
            var afterNorm = Norm(memory.MemoryVector);

            Assert.IsLessThan(beforeNorm, afterNorm, $"Memory norm should decrease after remove: before={beforeNorm}, after={afterNorm}");
        }

        // ---------------------------------------------------------------------------
        // Clear
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Clear_ZeroesMemoryVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateEntity("Opera");
            memory.Store(alice, likes, opera);

            memory.Clear();

            foreach (var v in memory.MemoryVector)
                Assert.AreEqual(0f, v);
        }

        [TestMethod]
        public void Clear_ZeroesFftMemoryVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var alice = memory.CreateEntity("Alice");
            var likes = memory.CreatePredicate("Likes");
            var opera = memory.CreateEntity("Opera");
            memory.Store(alice, likes, opera);

            memory.Clear();

            foreach (var v in memory.FftNormalizedMemoryVector)
                Assert.AreEqual(0f, v);
        }

        // ---------------------------------------------------------------------------
        // Dimension validation
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void Store_WrongDimensions_Throws()
        {
            var memory = new HolographicStorage(Dims, Id);
            var wrongVector = new float[Dims + 1];
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => memory.Store(wrongVector.AsSpan()));
        }

        [TestMethod]
        public void Remove_WrongDimensions_Throws()
        {
            var memory = new HolographicStorage(Dims, Id);
            var wrongVector = new float[Dims - 1];
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => memory.Remove(wrongVector.AsSpan()));
        }

        // ---------------------------------------------------------------------------
        // CreateSubject/Predicate/Object/Property with explicit value
        // ---------------------------------------------------------------------------

        [TestMethod]
        public void CreateSubject_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var customVec = new float[Dims];
            customVec[0] = 1f;

            var subject = memory.CreateEntity("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(subject.Vector.Span));
        }

        [TestMethod]
        public void CreatePredicate_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var customVec = new float[Dims];
            customVec[1] = 1f;

            var predicate = memory.CreatePredicate("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(predicate.Vector.Span));
        }

        [TestMethod]
        public void CreateObject_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var customVec = new float[Dims];
            customVec[2] = 1f;

            var obj = memory.CreateEntity("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(obj.Vector.Span));
        }

        [TestMethod]
        public void CreateProperty_WithExplicitValue_UsesProvidedVector()
        {
            var memory = new HolographicStorage(Dims, Id);

            var customVec = new float[Dims];
            customVec[3] = 1f;

            var prop = memory.CreateProperty("Custom", customVec.AsSpan());

            Assert.IsTrue(customVec.AsSpan().SequenceEqual(prop.Vector.Span));
        }

        [TestMethod]
        public void DeriveSubject_WithFactor_IsMultiplied()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a = memory.CreateEntity("Test");
            var b = a.Derive("float", 0.5f);

            Assert.AreEqual(a.Vector.Length, b.Vector.Length);

            var aSpan = a.Vector.Span;
            var bSpan = b.Vector.Span;
            for (var i = 0; i < a.Vector.Length; i++)
                Assert.AreEqual(aSpan[i] * 0.5f, bSpan[i]);
        }

        [TestMethod]
        public void DeriveSubject_WithFactor_IsMultiplied_Fft()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a = memory.CreateEntity("Test");
            var b = a.Derive("float", 0.5f);

            Assert.AreEqual(a.FftVector.Length, b.FftVector.Length);

            var aSpan = a.FftVector.Span;
            var bSpan = b.FftVector.Span;
            for (var i = 0; i < a.Vector.Length; i++)
                Assert.AreEqual(aSpan[i] * 0.5f, bSpan[i]);
        }

        [TestMethod]
        public void DerivePredicate_WithFactor_IsMultiplied()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a = memory.CreatePredicate("Test");
            var b = a.Derive("float", 0.5f);

            Assert.AreEqual(a.Vector.Length, b.Vector.Length);

            var aSpan = a.Vector.Span;
            var bSpan = b.Vector.Span;
            for (var i = 0; i < a.Vector.Length; i++)
                Assert.AreEqual(aSpan[i] * 0.5f, bSpan[i]);
        }

        [TestMethod]
        public void DeriveObject_WithFactor_IsMultiplied()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a = memory.CreateEntity("Test");
            var b = a.Derive("float", 0.5f);

            Assert.AreEqual(a.Vector.Length, b.Vector.Length);

            var aSpan = a.Vector.Span;
            var bSpan = b.Vector.Span;
            for (var i = 0; i < a.Vector.Length; i++)
                Assert.AreEqual(aSpan[i] * 0.5f, bSpan[i]);
        }

        [TestMethod]
        public void DeriveProperty_WithFactor_IsMultiplied()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a = memory.CreateProperty("Test");
            var b = a.Derive("float", 0.5f);

            Assert.AreEqual(a.Vector.Length, b.Vector.Length);

            var aSpan = a.Vector.Span;
            var bSpan = b.Vector.Span;
            for (var i = 0; i < a.Vector.Length; i++)
                Assert.AreEqual(aSpan[i] * 0.5f, bSpan[i]);
        }

        [TestMethod]
        public void Subject_Equals()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a1 = memory.CreateProperty("Test");
            var a2 = memory.CreateProperty("Test");
            var b = memory.CreateProperty("Foo");
            
            Assert.IsTrue(a1.Equals(a1));
            Assert.IsTrue(a1.Equals(a2));
            Assert.IsFalse(a1.Equals(b));
            Assert.IsFalse(a1.Equals(null));
            Assert.IsFalse(a1!.Equals(new object()));
            Assert.IsFalse(a1.Equals((object?)null));
            Assert.IsTrue(a1!.Equals((object?)a1));
            Assert.IsTrue(a1.Equals((object?)a2));
        }

        [TestMethod]
        public void Subject_InHashset_AddsAndRemoves()
        {
            var memory = new HolographicStorage(Dims, Id);

            var a1 = memory.CreateProperty("Test");
            var a2 = memory.CreateProperty("Test");
            var b = memory.CreateProperty("Foo");

            var set = new HashSet<MemoryProperty>();
            Assert.IsTrue(set.Add(a1));
            Assert.IsFalse(set.Add(a2));
            Assert.IsTrue(set.Add(b));

            Assert.IsTrue(set.Remove(a1));
            Assert.IsFalse(set.Remove(a2));
            Assert.IsTrue(set.Remove(b));
        }
    }
}
