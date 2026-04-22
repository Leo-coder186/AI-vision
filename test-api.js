async function test() {
    if (!process.env.MINIMAX_API_KEY) {
        throw new Error("Missing MINIMAX_API_KEY environment variable.");
    }

    try {
        const response = await fetch("https://api.minimaxi.com/anthropic/v1/messages", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "x-api-key": process.env.MINIMAX_API_KEY,
                "anthropic-version": "2023-06-01"
            },
            body: JSON.stringify({
                model: "MiniMax-M2.7",
                max_tokens: 1024,
                messages: [{ role: "user", content: "test" }]
            })
        });
        
        console.log("Status:", response.status);
        const data = await response.text();
        console.log("Response:", data);
    } catch (e) {
        console.error("Error:", e);
    }
}

test();
