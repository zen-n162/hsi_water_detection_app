export async function runInference(formData: FormData) {
  const res = await fetch("http://127.0.0.1:8000/api/inference/run", {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  return await res.json();
}
