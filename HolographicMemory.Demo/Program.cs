using HolographicMemory;
using static System.Numerics.Tensors.TensorPrimitives;

var memory = new HolographicMemory<float>(dimensions:8192, seed:13);

// People
var Martin = memory.CreateSubject("Martin");
var Alice = memory.CreateSubject("Alice");

// Relations
var Likes = memory.CreatePredicate("Likes");
//todo: var Dislikes = Likes.Select(a => -a).ToArray();
var Eats = memory.CreatePredicate("Eats");

// Topics
var Anime = memory.CreateObject("Anime");
var Programming = memory.CreateObject("Programming");
var Opera = memory.CreateObject("Opera");
var Pizza = memory.CreateObject("Pizza");

// Properties
var Man = memory.CreateProperty("Man");
var Woman = memory.CreateProperty("Man");

//todo: var animeHalf = anime.ToArray();
//todo: TensorPrimitives.Multiply(anime, 0.5f, animeHalf);

memory.Store(Martin, Man);
memory.Store(Alice, Woman);
memory.Store(Martin, Likes, Anime);
memory.Store(Martin, Likes, Programming);
memory.Store(Alice, Likes, Opera);
memory.Store(Alice, Eats, Pizza);
memory.Store(Martin, Eats, Pizza);

// Who likes anime?
var result = new float[memory.Dimensions];
memory.Query(Likes, Anime, result);
Console.WriteLine(Dot(result, Martin.Vector.Span));
Console.WriteLine(Dot(result, Alice.Vector.Span));

// What does Martin like?
var martinLikes = new float[memory.Dimensions];
memory.Query(Martin, Likes, martinLikes);

// decode (compare similarity)
Console.WriteLine(Dot(martinLikes, Anime.Vector.Span));
Console.WriteLine(Dot(martinLikes, Programming.Vector.Span));
Console.WriteLine(Dot(martinLikes, Opera.Vector.Span));

// What does Alice like?
var aliceLikes = new float[memory.Dimensions];
memory.Query(Alice, Likes, aliceLikes);

// decode (compare similarity)
Console.WriteLine(Dot(aliceLikes, Anime.Vector.Span));
Console.WriteLine(Dot(aliceLikes, Programming.Vector.Span));
Console.WriteLine(Dot(aliceLikes, Opera.Vector.Span));

// Remove fact
memory.Remove(Alice, Likes, Opera);

// What does Alice like?
var aliceLikes2 = new float[memory.Dimensions];
memory.Query(Alice, Likes, aliceLikes2);

// decode (compare similarity)
Console.WriteLine(Dot(aliceLikes2, Anime.Vector.Span));
Console.WriteLine(Dot(aliceLikes2, Programming.Vector.Span));
Console.WriteLine(Dot(aliceLikes2, Opera.Vector.Span));