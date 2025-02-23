
// document.addEventListener("DOMContentLoaded", () => {
//     const chatLog = document.getElementById("chat-log");
//     const userMessageInput = document.getElementById("user-message");
//     const sendButton = document.getElementById("send-btn");

//     // Function to add a message to the chat log
//     function addMessage(sender, message) {
//         const messageDiv = document.createElement("div");
//         messageDiv.classList.add(sender === "user" ? "user-message" : "bot-message");
//         messageDiv.textContent = message;
//         chatLog.appendChild(messageDiv);
//         chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
//     }

//     // Function to send user input to the backend
//     async function sendMessage() {
//         const userMessage = userMessageInput.value.trim();
//         if (userMessage === "") return;

//         // Display user message
//         addMessage("user", userMessage);
//         userMessageInput.value = ""; // Clear input field

//         try {
//             // Send message to Flask backend
//             const response = await fetch("http://localhost:5000/app", {
//                 method: "POST",
//                 headers: {
//                     "Content-Type": "application/json"
//                 },
//                 body: JSON.stringify({ message: userMessage })
//             });

//             const data = await response.json();
//             addMessage("bot", data.response); // Display bot response
//         } catch (error) {
//             console.error("Error:", error);
//             addMessage("bot", "Sorry, something went wrong. Please try again.");
//         }
//     }

//     // Event listener for the send button
//     sendButton.addEventListener("click", sendMessage);

//     // Allow pressing Enter to send a message
//     userMessageInput.addEventListener("keypress", (e) => {
//         if (e.key === "Enter") {
//             sendMessage();
//         }
//     });
// });

document.addEventListener("DOMContentLoaded", () => {
    const chatLog = document.getElementById("chat-log");
    const userMessageInput = document.getElementById("user-message");
    const sendButton = document.getElementById("send-btn");

    // Function to add a message to the chat log
    function addMessage(sender, message) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add(sender === "user" ? "user-message" : "bot-message");
      
        if (sender === "bot") {
          // Render HTML so the table is displayed properly
          messageDiv.innerHTML = message;
        } else {
          // Keep user messages as text
          messageDiv.textContent = message;
        }
      
        chatLog.appendChild(messageDiv);
        chatLog.scrollTop = chatLog.scrollHeight; // Scroll to the bottom
      }
      

    // Function to send user input to the backend
    async function sendMessage() {
        const userMessage = userMessageInput.value.trim();
        if (userMessage === "") return;

        // Display user message
        addMessage("user", userMessage);
        userMessageInput.value = ""; // Clear input field

        try {
            // Send message to Flask backend
            const response = await fetch("http://127.0.0.1:5000/chatbot", {  // Ensure the route is '/chatbot'
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({ message: userMessage }),
                mode: "cors"  // Explicitly set CORS mode
                
            });

            // Check if the response is OK (status 200-299)
            if (!response.ok) {
                throw new Error(`Server responded with status ${response.status}`);
            }

            const data = await response.json();
            addMessage("bot", data.response); // Display bot response
        } catch (error) {
            console.error("CORS or Network Error:", error);
            addMessage("bot", "Sorry, something went wrong. Please try again later.");
        }
    }

    // Event listener for the send button
    sendButton.addEventListener("click", sendMessage);

    // Allow pressing Enter to send a message
    userMessageInput.addEventListener("keypress", (e) => {
        if (e.key === "Enter") {
            sendMessage();
        }
    });
});
