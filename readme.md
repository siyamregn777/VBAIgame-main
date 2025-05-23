# Venture Builder AI

An interactive 3D office environment built with Python, PyGame, and OpenGL, featuring AI-powered NPCs.

## Prerequisites

- Python 3.8+
- OpenGL support
- PyGame

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/amplitudeventures/VBAIgame.git
   cd VBAIgame
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your OpenAI API key:
   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   ```

## Project Structure

```plaintext
venture-builder-ai/
├── textures/           # Generated texture files
│   ├── wall.png
│   ├── floor.png
│   └── ceiling.png
├── texture_generator.py # Texture generation script
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (not in repo)
└── README.md          # Project documentation
```

## Features

- 3D environment rendering using PyGame and OpenGL
- Procedurally generated textures
- AI-powered interactions using OpenAI API

## Usage

1. Generate textures (if not already present):
   ```bash
   python texture_generator.py
   ```

2. Run the main application:
   ```bash
   python app.py
   ```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for providing the AI capabilities
- PyGame community for the gaming framework
- OpenGL for 3D rendering support
```

