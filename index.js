import ollama from "ollama";

const sentence = `
My name is Subham Betal, and I'm a 23 years old Software Engineer with 3+ years of experience in web development. I specialize in React, Nextjs, and full-stack development, working on scalable web applications and SaaS products. You are currently pursuing a B.Tech in Electronics and Communication Engineering.`

/// Chunking the provided content
const chunks = sentence
  .split(".")
  .map((chunk) => chunk.trim())
  .filter(Boolean);

/// Generate embeddings
async function generateEmbeddings(chunks) {
  const response = await ollama.embed({
    model: "nomic-embed-text",
    input: chunks,
  });

  return response.embeddings.map((embedding, index) => ({
    text: chunks[index],
    embedding,
  }));
}

/// Calculate cosine similarity
function cosineSimilarity(a, b) {
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  if (normA === 0 || normB === 0) return 0;

  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

/// Similarity serach
async function similaritySearch(chunks, query, topK = 2) {

  const contentEmbeddings = await generateEmbeddings(chunks);

  const queryResult = await generateEmbeddings([query]);
  const queryEmbedding = queryResult[0].embedding;


  return contentEmbeddings
    .map((doc) => ({
      text: doc.text,
      score: cosineSimilarity(queryEmbedding, doc.embedding),
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

const question = "Education of Subham";

/// Test the search
const results = await similaritySearch(chunks, question)

const response = await ollama.chat({
  stream: true,
  model: "gemma3:4b",
  messages: [
    {
      role: "system",
      content:
        "Answer the user's question using only the provided context. If the answer is not in the context, say you don't have enough information. Give a clear, concise answer with proper questions.",
    },
    {
      role: "user",
      content: `
Question:
${question}

Retrieved context:
${results.map((result) => result.text).join("\n\n")}

Provide the answer:
      `.trim(),
    },
  ],
});


/// Stream the response
for await (const chunk of response) {
  process.stdout.write(chunk.message.content);
}

console.log();